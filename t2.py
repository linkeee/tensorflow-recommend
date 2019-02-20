import tensorflow as tf

state = tf.Variable(0, name='counter')
one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(5.0)

intermed = tf.add(input2, input3)
mul = tf.multiply(input1, intermed)

in1 = tf.placeholder(tf.float32)
in2 = tf.placeholder(tf.float32)
output = tf.multiply(in1, in2)

init_op = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init_op)
    # print(sess.run(state))
    for i in range(3):
        sess.run(update)
        # print(sess.run(state))

    print(sess.run([mul, intermed]))
    print(sess.run([output], feed_dict={in1: [7.], in2: [2.]}))

sess.close()
