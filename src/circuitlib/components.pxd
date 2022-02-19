# Create typedef to represent voltage/source signal functions
ctypedef double (*func)(double t, double value, double period, double mod, double x_offset, double y_offset) nogil

# TODO: Need to handle group 2 components using `group2_idx`
ctypedef struct comp:
    int type;  # type of component (resistor, capacitor etc.)
    int node1;  # the positive node
    int node2;  # the positive node
    double val1;
    double val2;  # TODO: work out if this is still needed
    double prev[2];  # voltage/current for component in previous time step
    func source;  # source signal (only relevant for V/I sourcses)
    double period;
    double mod;
    double x_offset;
    double y_offset;
