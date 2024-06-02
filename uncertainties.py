import math

# Direction
direction = False

# Given values
NoneNeg = -0.06945320218801498
NonePos = 0.2896474599838257
AddNeg = 0.2945330739021301
SubPos = -0.16713735461235046
SE_AddNeg = 0.010347
SE_NoneNeg = 0.014508
SE_NonePos = 0.013812
SE_SubPos = 0.009233

# Calculating NIE
if direction:
    NIE = (AddNeg - NoneNeg) / (NonePos - NoneNeg)
else:
    NIE = (SubPos - NonePos) / (NoneNeg - NonePos)

# Partial derivatives
if direction:
    partial_x = 1 / (NonePos - NoneNeg)
    partial_y = -(AddNeg - NonePos) / (NonePos - NoneNeg) ** 2
    partial_z = -(AddNeg - NoneNeg) / (NonePos - NoneNeg) ** 2
else:
    partial_x = 1 / (NoneNeg - NonePos)
    partial_y = -(SubPos - NoneNeg) / (NoneNeg - NonePos) ** 2
    partial_z = -(SubPos - NonePos) / (NoneNeg - NonePos) ** 2

# Propagated uncertainty
if direction:
    sigma_NIE = math.sqrt(
        (partial_x * SE_AddNeg) ** 2 +
        (partial_y * SE_NoneNeg) ** 2 +
        (partial_z * SE_NonePos) ** 2
    )
else:
    sigma_NIE = math.sqrt(
        (partial_x * SE_SubPos) ** 2 +
        (partial_y * SE_NonePos) ** 2 +
        (partial_z * SE_NoneNeg) ** 2
    )

print("Direction:", direction)
print("NIE:", round(NIE, 2))
print("Uncertainty:", round(sigma_NIE, 2))
