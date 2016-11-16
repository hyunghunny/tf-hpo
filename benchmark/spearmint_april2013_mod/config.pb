language: PYTHON
name:     "HPOlib.cv"

variable {
 name: "FILTER_SIZE"
 type: INT
 size: 1
 min:  1
 max:  14
}

variable {
 name: "CONV1_DEPTH"
 type: INT
 size: 1
 min:  1
 max:  512
}

variable {
 name: "CONV2_DEPTH"
 type: INT
 size: 1
 min:  1
 max:  512
}

variable {
 name: "FC1_WIDTH"
 type: INT
 size: 1
 min:  1
 max:  1024
}

variable {
 name: "BASE_LEARNING_RATE"
 type: FLOAT
 size: 1
 min:  0.0001
 max:  0.1
}

variable {
 name: "DROPOUT_RATE"
 type: FLOAT
 size: 1
 min:  0.1
 max:  1
}

variable {
 name: "REGULARIZER_FACTOR"
 type: FLOAT
 size: 1
 min:  0
 max:  1
}
