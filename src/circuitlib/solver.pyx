import cython
from cython.parallel import prange
import numpy as np
cimport numpy as np

from libc.stdlib cimport malloc, calloc, free
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free, PyMem_RawRealloc
from libc.math cimport fabs, isnan, M_PI
from libc.math cimport sin as csin
from scipy.linalg.cython_lapack cimport dgesv # TODO: is there a sparse cython_lapack?

from libc.stdint cimport uintptr_t
from .signals cimport DC, sin, sawtooth, square
from .components cimport comp

cpdef dae(double[::1] time, double[::1,:] a1, double[::1,:] a2, double[::1] init, comp_dict):
    """Solves the differential algebriac equation of the circuits transient
    response.

    Args:
        a1:
        a2:
        init:
        end:
        h_n:
        comp_dict:
    """
    # Populate struct with component values
    cdef size_t n_components = len(comp_dict)
    # TODO: keep an array of components which we use for stamping the RSM, which
    # we then solve for.
    cdef comp * components = <comp *>malloc(n_components * sizeof(comp))
    if not (components):
        raise MemoryError()
    cdef size_t matrix_shape = a1.shape[0]
    # Parse comopnent dict into struct and return the maximum expected voltage
    # for the nodes of the circuit.
    # TODO: come up with better struct format for the components (and we need to
    # rename these to elements)
    cdef double voltage = parse_components(comp_dict, components, matrix_shape)

    # Setting up variables for LAPACK solver
    cdef int n=a1.shape[0], nrhs=1, lda=a1.shape[0], ldx=init.shape[0], ldaf=a1.shape[0], ldx_n1=init.shape[0], info
    cdef int *ipiv = <int *>malloc(n * sizeof(int))
    if not ipiv:
        raise MemoryError()

    # Arrays to hold solutions to the MNA at a given time step
    # TODO: clearer naming convention for all of these arrays. this need not be
    # similar to how they are referred to in the textbook, but what is clearest
    # TODO: could we put the next two blocks of code into a separate function?
    cdef np.ndarray[double, ndim=2] A = np.zeros((n,n), dtype=np.double, order="F")
    cdef np.ndarray[double, ndim=2] x_n = np.zeros((n,2), dtype=np.double, order="F")
    cdef np.ndarray[double, ndim=2] x_n1 = np.empty((ldx,2), dtype=np.double, order="F")
    cdef np.ndarray[double, ndim=2] x_n2 = np.empty((ldx,2), dtype=np.double, order="F")
    cdef np.ndarray[double, ndim=2] x_tmp = np.zeros((ldx,2), dtype=np.double, order="F")
    cdef np.ndarray[double, ndim=2] v = np.zeros((time.shape[0],ldx), dtype=np.double, order="C")
    # cdef np.ndarray[double, ndim=1] time_tracker = np.zeros(time.shape[0], dtype=np.double, order="C")

    # Views for arrays to improve speed
    cdef double [::1,:] x_n_view = x_n
    cdef double [::1,:] x_n1_view = x_n1
    cdef double [::1,:] x_n2_view = x_n2
    cdef double [::1,:] x_tmp_view = x_tmp
    cdef double [::1,:] a1_view=a1
    cdef double [::1,:] a2_view=a2
    cdef double [::1,:] A_view=A
    cdef double [::] t_view = time
    cdef double [:,::1] v_view = v
    v_view[0,:] = init  # TODO: need clearer name for the `init` matrix

    # Copy initial conditions into x_n array
    # TODO: Do we include the results of the initial conditions in the
    # results array?
    for i in range(init.shape[0]):
        x_n[i,0] = init[i]
        x_n[i,1] = init[i]
        x_n1[i,0] = init[i]
        x_n1[i,1] = init[i]
        x_tmp[i,0] = init[i]
        x_tmp[i,1] = init[i]

    # SETTING UP VARIABLES FOR INTEGRATION
    cdef double start = time[0]
    cdef double end = time[-1]
    # Measurements must be equal or smaller than the steps in the supplied time array
    cdef double max_step = time[1] - time[0]
    cdef double h_n = min(max_step, (time[-1] - time[0]) / 1000)  # TODO: won't this always be the second variable?
    cdef double h_tmp=h_n, h_n0=h_n, h_n1=h_n, h_n_tmp=h_n, h_n0_tmp=h_n, h_ # previous prevous step sizes
    cdef double t=start, t_tmp=start  # the first step after initialisation
    # Assume the solver will complete the simulation in 100 times the number
    # of steps given by the time span and initial step size.
    cdef int total_success_steps = time.shape[0], max_steps = total_success_steps * 100

    cdef double plte=0.0, error=0.0, alpha, p=1       # error of integration
    cdef int step=-1, succesful_steps=1, j            # count iterations
    cdef double i_n=0.0, i_n0, i_, v_                 # currents/voltages at the present step
    cdef int add_next_step=0                          # always save result at t=start

    try:
        # Iterate through time steps and solve for the currents/voltages passing
        # through the circuit at the _next_ time step until reaching the end
        # of the transient period.
        # while t <= end:
        while succesful_steps < total_success_steps:

            step += 1

            # We use Richardson extrapolation, meaning we must solve for the
            # next step twice under different time steps to estimate the local
            # truncation error. With this we can determine if we should accept
            # or increase/decrease the time step.
            for j in range(2):
                # Ignore first step as voltages have already been solved for
                # using the initial DC conditions
                if step==0:
                    continue

                if j==0:
                    h_ = h_n  # use current time step length
                else:
                    # Use the current step plus the previous step length
                    h_ = h_n + h_n0

                # Create MNA stamps and solve for x
                solveStep(x_n1_view, x_n_view, components, n_components, t, h_, j,
                          a1_view, a2_view, A_view, n, nrhs, lda, ipiv, ldx, info)

            # Calculate error between the two solutions
            # TODO: definitely need better error measurements
            alpha = h_n0 / h_n
            plte = 0.0
            for i in range(3):
                error = ((x_n1_view[i,0] - x_n1_view[i,1])
                          / ((1.0 + alpha)**(p+1) - 1.0))
                error /= voltage  # normalise error by the source's voltage
                # Take the largest absolute error of all the nodes
                if fabs(error) > plte:
                    plte = fabs(error)

            if (step < 2) or (1e-6 <= plte <= 1e-1) or (plte == 0) or (h_n == max_step and plte <= 1e-1):
                # Step succesful. Increment steps and save results.

                # Checking if current time step solved for is past the next
                # checkpoint. If so, solve for that next time step before moving
                # forward.
                step_overshoot = ((t+h_n) - t_view[succesful_steps]) / h_n

                # Solving for the next checkpoint in the time array before
                # moving forward with the transient analysis
                if 0 <= step_overshoot < 0.5:
                    j = 0
                    h_ = t_view[succesful_steps] - t
                    solveStep(x_tmp_view, x_n_view, components, n_components, t, h_, j,
                            a1_view, a2_view, A_view, n, nrhs, lda, ipiv, ldx, info)
                    addResults(v_view[succesful_steps,:], x_tmp_view[:,j])
                    succesful_steps += 1
                elif step_overshoot >= 0.5:
                    # Solve for x using the previous solution. This prevents
                    # errors when using a very small value for h_n.
                    j = 1
                    t_tmp = t - h_n0  # TODO: check if we want to use h_n0 not h_n. I think this is right...
                    h_ = t_view[succesful_steps] - t_tmp
                    solveStep(x_tmp_view, x_n_view, components, n_components, t_tmp, h_, j,
                            a1_view, a2_view, A_view, n, nrhs, lda, ipiv, ldx, info)
                    addResults(v_view[succesful_steps,:], x_tmp_view[:,j])
                    succesful_steps += 1

                update_prev(x_n1_view[:,0], x_n_view[:,0], components, n_components, h_n)
                update_xs(x_n1_view, x_n_view)
                h_n0 = h_n

                t += h_n

            elif (0 < plte < 1e-6):
                # plte uneccesarily small. Increase step size and repeat step.
                h_n = min(h_n*2, max_step)
            elif plte > 1e-1:
                # plte too large. Decrease step size and repeat step.
                h_n = h_n / 2.0

            if step > max_steps:  # TODO: raise error if over 50% of steps unsuccesful
                print("too many steps!", step)
                break
        return v  # TODO: does this even need to return anything? I think the matrix will be updated in place...

    finally:
        PyMem_Free(ipiv)
        PyMem_Free(components)

    # TODO: Check if the full time range has been simulated. Perhaps do a sense check
    # at the start to ensure appropriate initial starting step size, then potentially
    # rerun the simulation again if it has failed in such a manner.

cdef parse_components(comp_dict, comp* comp_list, matrix_shape):
    cdef double circuit_voltage

    for i, val in enumerate(comp_dict.values()):
        comp_list[i].node1 = val["nodes"][0] - 1
        comp_list[i].node2 = val["nodes"][1] - 1
        comp_list[i].prev[0] = 0
        comp_list[i].prev[1] = 0
        comp_list[i].val1 = val["value"]
        comp_list[i].val2 = 0

        if val["type"] == "V":
            circuit_voltage = val["value"]
            comp_list[i].node1 = val["group2_idx"]
            comp_list[i].type = 0
            comp_list[i].period = val["set_kwargs"]["period"]
            comp_list[i].x_offset = val["set_kwargs"]["x_offset"]
            comp_list[i].y_offset = val["set_kwargs"]["y_offset"]
            comp_list[i].mod = val["set_kwargs"]["mod"]
            # TODO: how are we auto allocating the signal generator?
            # Could use big if/elif statement like below but it's verbose.
            # TODO: Currently just testing with DC voltage step changes
            comp_list[i].source = DC
            # if val["signal"].__name__ == "DC":
            #     comp_list[i].source = DC
            # elif val["signal"].__name__ == "sin":
            #     comp_list[i].source = sin
            # elif val["signal"].__name__ == "square":
            #     comp_list[i].source = square
            # elif val["signal"].__name__ == "sawtooth":
            #     comp_list[i].source = sawtooth

        if val["type"] == "R":
            comp_list[i].type = 1

        if val["type"] == "C":
            comp_list[i].type = 2

        if val["type"] == "L":
            comp_list[i].node1 = val["group2_idx"]
            comp_list[i].type = 3
    return circuit_voltage

cdef void solveStep(double[::1,:] x, double[::1,:] x_prev, comp* components,
                    int n_components, double t, double step, int j, double[::1,:] a1,
                    double[::1,:] a2, double[::1,:] out, int n, int nrhs, int lda,
                    int* ipiv, int ldx, int info) nogil:
    """Updates the x arrays with the solution to the next time step."""
    fillZeros(x[:,j])  # remove previous values stored in x
    stamp(x[:,j], x_prev[:,j], components, n_components, t, step, j)
    addDivide(a1, a2, step, out)
    # TODO: Check info from dgesv to ensure it succesfully completed
    dgesv(&n, &nrhs, &out[0,0], &lda, &ipiv[0], &x[0,j], &ldx, &info)

cdef int stamp(double[::1] x, double[::1] x_prev, comp* c, int n, double t, double step, int it) nogil:
    """Stamp transient matrix with next values.

    Args:
        x (array): the current values of x.
        x_prev (array): the previous values of x.
        c: Array of circuit components, each defined as `comp` structs.
        n: Number of circuit components.
        t: The current time step.
        step: The length of the time step.
        it: TODO
    """

    for i in range(n):

        if c[i].type == 0:  # if voltage source
            x[c[i].node1] += c[i].source(t, c[i].val1, c[i].period, c[i].mod, c[i].x_offset, c[i].y_offset)

        # TODO: Create a resource in the docs that explains these equations below
        # TODO: does the below work if it is a group 2 capacitor?
        elif c[i].type == 2:  # if capacitor
            if c[i].node1 == -1:  # if node1 connected to ground
                x[c[i].node2] += c[i].prev[it] + (2.0*c[i].val1/step)*x_prev[c[i].node2]
            elif c[i].node2 == -1:  # if node2 connected to ground
                x[c[i].node1] += c[i].prev[it] + (2.0*c[i].val1/step)*x_prev[c[i].node1]
            else:
                x[c[i].node1] += (c[i].prev[it] + (2.0*c[i].val1/step)*(x_prev[c[i].node1] - x_prev[c[i].node2]))
                x[c[i].node2] -= (c[i].prev[it] + (2.0*c[i].val1/step)*(x_prev[c[i].node1] - x_prev[c[i].node2]))
        elif c[i].type == 3:  # if inductor
            x[c[i].node1] += - (2.0*c[i].val1/step)*x_prev[c[i].node1] - c[i].prev[it]

    return n


cdef void update_prev(double[::1] x, double[::1] x_prev, comp* c, int n, double step) nogil:
    """Define the new x values."""
    # Iterate through each component and stamp value onto x
    cdef double u = 0, u_prev = 0, i = 0, i_prev = 0
    for idx in range(n):
        c[idx].prev[1] = c[idx].prev[0]

        if c[idx].type == 2:  # if capacitor
            # TODO: This will get very bloated very quick! Need to think of a better way for this.
            if c[idx].node2 == -1:
                u = x[c[idx].node1]
                u_prev = x_prev[c[idx].node1]
            elif c[idx].node1 == -1:
                u = x[c[idx].node2]
                u_prev = x_prev[c[idx].node2]
            else:
                u = x[c[idx].node1] - x[c[idx].node2]
                u_prev = x_prev[c[idx].node1] - x_prev[c[idx].node2]
            c[idx].prev[0] = ((2.0*c[idx].val1/step)*u - (c[idx].prev[1] + ((2.0*c[idx].val1/step)*u_prev)))

        if c[idx].type == 3:  # if inductor
            i = x[c[idx].node1]
            i_prev = x_prev[c[idx].node1]
            c[idx].prev[0] = ((2.0*c[idx].val1/step)*i - (c[idx].prev[1] + ((2.0*c[idx].val1/step)*i_prev)))


cdef void fillZeros(double[::1] x) nogil:
    """Convenience function to set all entries of an array to zero."""
    cdef int dim=x.shape[0]
    for i in range(dim):
        x[i] = 0

cdef void copy_array(double[::1,:] x1, double[::1,:] x2) nogil:
    """Copy all values from x2 into x1."""
    cdef int dim1=x1.shape[0]
    cdef int dim2=x1.shape[1]
    for i in range(dim1):
        for j in range(dim2):
            x1[i,j] = x2[i,j]

cdef void update_xs(double[::1,:] x_new, double[::1,:] x) nogil:
    """Copy values from new x values into x array."""
    cdef int dim=x.shape[0]
    for i in range(dim):
        x[i,1] = x[i,0]
        x[i,0] = x_new[i,0]


cdef void addResults(double[::1] results, double[::1] x) nogil:
    """Copy values from x array into the growing list of results."""
    cdef int dim=x.shape[0]
    for i in range(dim):
        results[i] = x[i]

cdef void addResults2(double * results, double[::1] x) nogil:
    """Copy values from x array into the growing list of results."""
    cdef int dim=x.shape[0]
    for i in range(dim):
        results[i] = x[i]


cdef void addDivide(double[::1,:] a1, double[::1,:] a2, double h, double[::1,:] out) nogil:
    """Combine results from group 1 and group 2 matrices."""
    cdef int dim1=a1.shape[0], dim2=a1.shape[1], idx, result=0
    for i in range(dim1):
        for j in range(dim2):
            out[i,j] = a1[i,j] + (a2[i,j] / h)


