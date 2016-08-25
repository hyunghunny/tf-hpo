language: PYTHON
name:     "HPOlib.cv"

variable {
 name: "filter_size"
 type: INT
 size: 1
 min:  1
 max:  28
}

variable {
 name: "conv1_depth"
 type: INT
 size: 1
 min:  1
 max:  512
}

variable {
 name: "conv2_depth"
 type: INT
 size: 1
 min:  1
 max:  512
}

variable {
 name: "fc_depth"
 type: INT
 size: 1
 min:  1
 max:  512
}