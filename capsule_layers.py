import tensorflow as tf


class ConvCapsuleLayer(tf.keras.layers.Layer):
    def __init__(self, num_capsules, capsule_dim, kernel_size, strides=1, padding='valid'):
        super(ConvCapsuleLayer, self).__init__()
        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.conv = tf.keras.layers.Conv2D(
            filters=num_capsules*capsule_dim,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            activation=None,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros'
        )

    def call(self, inputs):
        # Perform convolution
        outputs = self.conv(inputs)

        # Reshape the output tensor
        shape = tf.shape(outputs)
        num_capsules = self.num_capsules
        capsule_dim = self.capsule_dim
        outputs = tf.reshape(outputs, (shape[0], shape[1], shape[2], num_capsules, capsule_dim))

        # Apply squash activation
        norm = tf.norm(outputs, axis=-1, keepdims=True)
        norm_squared = norm ** 2
        outputs = outputs * norm_squared / (1 + norm_squared) / norm

        return outputs


class DenseCapsuleLayer(tf.keras.layers.Layer):
    def __init__(self, num_capsules, capsule_dim):
        super(DenseCapsuleLayer, self).__init__()
        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim
        self.dense = tf.keras.layers.Dense(
            units=num_capsules*capsule_dim,
            activation=None,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros'
        )

    def call(self, inputs):
        # Flatten the input tensor
        shape = tf.shape(inputs)
        inputs = tf.reshape(inputs, (shape[0], -1))

        # Perform dense layer transformation
        outputs = self.dense(inputs)

        # Reshape the output tensor
        num_capsules = self.num_capsules
        capsule_dim = self.capsule_dim
        outputs = tf.reshape(outputs, (shape[0], num_capsules, capsule_dim))

        # Apply squash activation
        norm = tf.norm(outputs, axis=-1, keepdims=True)
        norm_squared = norm ** 2
        outputs = outputs * norm_squared / (1 + norm_squared) / norm

        return outputs


class Length(tf.keras.layers.Layer):
    def call(self, inputs):
        # Compute the norm of the input tensor
        return tf.norm(inputs, axis=-1)
