import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


class FGSM_Attack:
	def __init__(self, model, images, labels, epsilons, target=None):
		self.model = model
		self.images = images
		self.labels = labels
		self.epsilons = epsilons
		self.target = target

		self.loss_object = tf.keras.losses.CategoricalCrossentropy()
		self.adv_examples = {}

	@staticmethod
	def decode_prediction(prediction):
		min_conf = -1
		result_id = None
		result_conf = None
		for index, conf in enumerate(prediction[0]):
			if conf > min_conf:
				result_id = index
				result_conf = conf
				min_conf = conf
		return result_id, result_conf

	@staticmethod
	def perturb(image, eps, signed_grad, target):
		if target:
			adv = (image - (signed_grad * eps)).numpy()
		else:
			adv = (image + (signed_grad * eps)).numpy()

		return adv

	def run(self):
		for eps in self.epsilons:
			self.adv_examples[eps] = []
			successful_attacks = 0

			for image, label in zip(self.images, self.labels):
				is_success = False

				# preprocess
				image = tf.cast(image, tf.float32)
				image = image[None, ...]
				label = np.reshape(label, (1, 10))

				# gradient
				with tf.GradientTape() as tape:
					tape.watch(image)
					prediction = self.model(image)
					if self.target is not None:
						target_label = np.zeros(shape=(1, 10))
						target_label[0][self.target] = 1
						loss = self.loss_object(target_label, prediction)
					else:
						loss = self.loss_object(label, prediction)
				gradient = tape.gradient(loss, image)
				signed_grad = tf.sign(gradient)

				# perturb
				adversary = self.perturb(image, eps, signed_grad, self.target)
				cls_orig, conf_orig = self.decode_prediction(label)
				cls_adv, conf_adv = self.decode_prediction(self.model(adversary))
				if self.target:
					if cls_adv == self.target and cls_orig != self.target:
						is_success = True
				else:
					if cls_adv != cls_orig:
						is_success = True

				if is_success:
					successful_attacks += 1
					if len(self.adv_examples[eps]) < 5:
						adversary = adversary.squeeze()
						self.adv_examples[eps].append((cls_orig, cls_adv, adversary))

			success_rate = successful_attacks / float(len(self.images))
			print("Epsilon: {}\tAttack Success Rate = {} / {} = {:.2f}".format(eps, successful_attacks,
																			   len(self.images), success_rate))

	def visualize(self):
		plt.figure(figsize=(8, 10))
		cnt = 0
		for eps, adv_examples in self.adv_examples.items():
			for index, data in enumerate(adv_examples):
				cnt += 1
				plt.subplot(len(self.adv_examples.keys()), len(adv_examples), cnt)
				plt.xticks([], [])
				plt.yticks([], [])
				if index == 0:
					plt.ylabel("Eps: {}".format(eps), fontsize=14)

				orig, adv, adv_ex = data
				plt.title("{} -> {}".format(orig, adv))

				plt.imshow(adv_ex, cmap="gray")
			cnt += 4
			cnt -= cnt % 5
		plt.tight_layout()
		plt.show()
