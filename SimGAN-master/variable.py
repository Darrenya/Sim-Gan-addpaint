import tensorflow as tf
state = tf.Variable(0,name='counter')
# print(state.name)
one = tf.constant(1)

new_value = tf.add(state , one)
update = tf.assign(state,new_value)#new_value的值给state

init = tf.initialize_all_variables()#定义变量的话一定得有这个
with tf.Session() as sess:
	sess.run(init)
	for _ in range(3):
		sess.run(update)
		print(sess.run(state))