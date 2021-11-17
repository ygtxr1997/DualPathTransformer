import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-0.15, 3.2, num=100)

""" 1. loss """
cosine = np.cos(x)
arcface = np.cos(x + 0.5)
cosface = np.cos(x) - 0.35
linearface = -0.88 * x + 0.88

a = 1.2
k = 0.1
lmcosface = np.cos(x) - 0.35 + k * (x - a)
lmarcface = np.cos(x + 0.5 - k * (x - a))

plt.figure()
plt.plot(x, cosine, 'b--', label=r'Softmax')
plt.plot(x, arcface, 'g:', label=r'ArcFace:cos(x+0.5)')
plt.plot(x, cosface, 'r:', label=r'CosFace:cos(x)-0.35')
plt.plot(x, linearface, 'c:', label=r'LinearFace:-0.88x+0.88')
plt.plot(x, lmcosface, 'y-', label=r'LM-CosFace:$m_2$=0.35,k=0.1,a=1.2')
plt.plot(x, lmarcface, 'm-', label=r'LM-ArcFace:$m_1$=0.5,k=0.1,a=1.2')

# plt.title('prob.')
plt.legend(loc='lower left',)
plt.grid()

plt.ylabel("The value of logit term")
plt.xlabel(r"Angle radian $\theta$ between a feature and its class center")

plt.savefig('loss-prob.pdf')
plt.clf()

""" 2. margin in logit space """
arcface_margin = cosine - arcface
cosface_margin = cosine - cosface
linearface_margin = cosine - linearface
lmcosface_margin = cosine - lmcosface
lmarcface_margin = cosine - lmarcface

plt.figure(figsize=(6, 2.4))
plt.subplots_adjust(bottom=0.2)
plt.plot(x, cosine - cosine, 'b--', label='Softmax')
# plt.plot(x, arcface_margin, 'g:', label='ArcFace:cos(x+0.5)')
plt.plot(x, cosface_margin, 'r:', label='CosFace:cos(x)-0.35')
# plt.plot(x, linearface_margin, 'c:', label='LinearFace:-0.88x+0.88')
plt.plot(x, lmcosface_margin, 'y-', label='LM-CosFace:$m_2$=0.35,k=0.1,a=1.2')
# plt.plot(x, lmarcface_margin, 'm-', label='LM-ArcFace:cos(x+0.5-k(x-a)')

# plt.title('logit margin')
plt.legend(loc='lower left', bbox_to_anchor=(0., 0.05))
plt.grid()

plt.ylabel("Margin in logit space")
plt.xlabel(r"Angle radian $\theta$ between a feature and its class center")

plt.savefig('loss-margin-logit.pdf')
plt.clf()


""" 3. margin in angular space """
arcface_angular_margin = (x + 0.5) - x
# cosface_angular_margin = x - np.arccos(cosface)
# linearface_angular_margin = x - np.arccos(linearface)
lmarcface_angular_margin = (x + 0.5 - k * (x - a)) - x
# lmcosface_angular_margin = x - np.arccos(lmcosface)

plt.figure(figsize=(6, 2.4))
plt.subplots_adjust(bottom=0.2)
plt.plot(x, x - x, 'b--', label='Softmax')
plt.plot(x, arcface_angular_margin, 'g:', label='ArcFace:cos(x+0.5)')
# plt.plot(x, cosface_angular_margin, 'r:', label='CosFace:cos(x)-0.35')
# plt.plot(x, linearface_angular_margin, 'c:', label='LinearFace:-0.88x+0.88')
# plt.plot(x, lmcosface_margin, 'y-', label='LM-CosFace:cos(x)-0.35+k(x-a)')
plt.plot(x, lmarcface_angular_margin, 'm-', label='LM-ArcFace:$m_1$=0.5,k=0.1,a=1.2')

# plt.title('angular_margin')
plt.legend(loc='lower left', bbox_to_anchor=(0., 0.05))
plt.grid()

plt.ylabel("Margin in angular space")
plt.xlabel(r"Angle radian $\theta$ between a feature and its class center")

plt.savefig('loss-margin-angular.pdf')
plt.clf()