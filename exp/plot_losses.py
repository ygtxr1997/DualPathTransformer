import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(27, 9))

x = np.linspace(-0.15, 3.2, num=100)

""" 1. loss """
cosine = np.cos(x)
arcface = np.cos(x + 0.5)
cosface = np.cos(x) - 0.4
linearface = -0.88 * x + 0.88

# a = 0.12
# b = 2.6
# c = 1.6
# quadraticface = -a * (x + b) * (x + b) + c

a = 1.2
k = 0.1
adacosface = np.cos(x) - 0.4 + k * (x - a)

plt.subplot(131)
plt.plot(x, cosine, 'b-', label='Softmax:cos(x)')
plt.plot(x, arcface, 'g-', label='ArcFace:cos(x+0.5)')
plt.plot(x, cosface, 'r-', label='CosFace:cos(x)-0.4')
plt.plot(x, linearface, 'c-', label='LinearFace:-0.88x+0.88')
# plt.plot(x, quadraticface, 'y--', label='QuadraticFace:-'+str(a)+'(x+'+str(b)+')^2+'+str(c))
plt.plot(x, adacosface, 'y--', label='AdaCosFace:cos(x)-0.4+k(x-a)')

plt.title('prob.')
plt.legend(loc='upper right')
plt.grid()


""" 2. margin """
arcface_margin = cosine - arcface
cosface_margin = cosine - cosface
linearface_margin = cosine - linearface
# quadraticface_margin = cosine - quadraticface
adacosface_margin = cosine - adacosface

plt.subplot(132)
plt.plot(x, cosine - cosine, 'b-', label='Softmax:cos(x)')
plt.plot(x, arcface_margin, 'g-', label='ArcFace:cos(x+0.5)')
plt.plot(x, cosface_margin, 'r-', label='CosFace:cos(x)-0.4')
plt.plot(x, linearface_margin, 'c-', label='LinearFace:-0.88x+0.88')
# plt.plot(x, quadraticface_margin, 'y--', label='QuadraticFace:-'+str(a)+'(x+'+str(b)+')^2+'+str(c))
plt.plot(x, adacosface_margin, 'y--', label='AdaCosFace:cos(x)-0.4+k(x-a)')

plt.title('margin')
plt.legend(loc='upper right')
plt.grid()

plt.ylabel("Margin on logit space")
plt.xlabel("Angle radian $\theta$ between a feature and its class center")


""" 3. grad """
cosine_grad = -np.sin(x)
arcface_grad = -np.sin(x + 0.5)
cosface_grad = -np.sin(x)
linearface_grad = 0.0 * x - 0.88
# quadraticface_grad = -a * 2 * (x + b)
adacosface_grad = -np.sin(x) + k

plt.subplot(133)
plt.plot(x, -cosine_grad, 'b-', label='Softmax:cos(x)')
plt.plot(x, -arcface_grad, 'g-', label='ArcFace:cos(x+0.5)')
plt.plot(x, -cosface_grad, 'r-', label='CosFace:cos(x)-0.4')
plt.plot(x, -linearface_grad, 'c-', label='LinearFace:-0.88x+0.88')
# plt.plot(x, -quadraticface_grad, 'y--', label='QuadraticFace:-'+str(a)+'(x+'+str(b)+')^2+'+str(c))
plt.plot(x, -adacosface_grad, 'y--', label='AdaCosFace:cos(x)-0.4+k(x-a)')

plt.title('grad')
plt.legend(loc='upper right')
plt.grid()

plt.ylabel("Margin on angular space")
plt.xlabel("Angle radian $\theta$ between a feature and its class center")

"""" End. Save """
plt.savefig('losses.pdf')