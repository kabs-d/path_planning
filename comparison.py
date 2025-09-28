import matplotlib.pyplot as plt

# Modeled y-values
modeled_y = [
    1263.77, 962.20, 831.31, 758.16, 711.45, 679.04, 655.23, 637.01,
    622.60, 610.93, 601.29, 593.18, 586.27, 580.32, 575.13, 570.56,
    566.52, 562.91, 559.68, 556.76, 554.11
]

# Main projected points (first 21 y-values from your projected points)
main_param_y = [
    1264, 962.5, 832.5, 760.5, 714.5, 683, 659, 641, 627, 616, 606, 598, 592, 586, 581, 576, 572, 569, 566, 563, 560
]

# Extra x-values (reversed)
x111_y = list(reversed([
    560, 563, 566, 569, 572, 576, 581, 586, 592, 598,
    606, 616, 627, 641, 659, 683, 715, 761, 833, 963, 1266
]))

xm57_y = list(reversed([
    560, 563, 566, 569, 572, 576, 581, 586, 592, 598,
    606, 616, 627, 641, 659, 682, 714, 760, 832, 962, 1262
]))

x168_y = list(reversed([
    560, 563, 566, 569, 572, 576, 581, 586, 592, 599,
    606, 616, 627, 642, 660, 683, 715, 761, 834, 964, 1267
]))

xm114_y = list(reversed([
    560, 563, 565, 569, 572, 576, 581, 586, 591, 598,
    606, 616, 627, 641, 659, 682, 714, 760, 832, 961, 1260
]))

# X-axis (point indices)
x = range(1, 22)  # 1 to 21

plt.figure(figsize=(12,6))
plt.plot(x, modeled_y, 'o-', label='Modeled', linewidth=2)
plt.plot(x, main_param_y, 's-', label='Main Param', linewidth=2)
plt.plot(x, x111_y, '^-', label='Extra x=111', linewidth=2)
plt.plot(x, xm57_y, 'v-', label='Extra x=-57', linewidth=2)
plt.plot(x, x168_y, 'd-', label='Extra x=168', linewidth=2)
plt.plot(x, xm114_y, 'p-', label='Extra x=-114', linewidth=2)

plt.xlabel('Point index')
plt.ylabel('Y pixel value')
plt.title('Comparison of Modeled vs Projected Points (Reversed Extra Lists)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
