#!/usr/bin/env python
# coding: utf-8

# [Table of Contents](./table_of_contents.ipynb)

# # Symbols and Notations
#
# Here is a collection of the notation used by various authors for the linear Kalman filter equations.

# ## Labbe
#
#
# $$
# \begin{aligned}
# \overline{\mathbf x} &= \mathbf{Fx} + \mathbf{Bu} \\
# \overline{\mathbf P} &=  \mathbf{FPF}^\mathsf{T} + \mathbf Q \\ \\
# \mathbf y &= \mathbf z - \mathbf{H}\overline{\mathbf x} \\
# \mathbf S &= \mathbf{H}\overline{\mathbf P}\mathbf{H}^\mathsf{T} + \mathbf R \\
# \mathbf K &= \overline{\mathbf P}\mathbf{H}^\mathsf{T}\mathbf{S}^{-1} \\
# \mathbf x  &= \overline{\mathbf x} +\mathbf{Ky} \\
# \mathbf P &= (\mathbf{I}-\mathbf{KH})\overline{\mathbf P}
# \end{aligned}$$
#
#
# ## Wikipedia
# $$
# \begin{aligned}
# \hat{\mathbf x}_{k\mid k-1} &= \mathbf{F}_{k}\hat{\mathbf x}_{k-1\mid k-1} + \mathbf{B}_{k} \mathbf{u}_{k} \\
# \mathbf P_{k\mid k-1} &=  \mathbf{F}_{k} \mathbf P_{k-1\mid k-1} \mathbf{F}_{k}^{\textsf{T}} + \mathbf Q_{k}\\
# \tilde{\mathbf{y}}_k &= \mathbf{z}_k - \mathbf{H}_k\hat{\mathbf x}_{k\mid k-1} \\
# \mathbf{S}_k &= \mathbf{H}_k \mathbf P_{k\mid k-1} \mathbf{H}_k^\textsf{T} + \mathbf{R}_k \\
# \mathbf{K}_k &= \mathbf P_{k\mid k-1}\mathbf{H}_k^\textsf{T}\mathbf{S}_k^{-1} \\
# \hat{\mathbf x}_{k\mid k} &= \hat{\mathbf x}_{k\mid k-1} + \mathbf{K}_k\tilde{\mathbf{y}}_k \\
# \mathbf P_{k|k} &= (I - \mathbf{K}_k \mathbf{H}_k) \mathbf P_{k|k-1}
# \end{aligned}$$
#
# ## Brookner
#
# $$
# \begin{aligned}
# X^*_{n+1,n} &= \Phi X^*_{n,n} \\
# X^*_{n,n}  &= X^*_{n,n-1} +H_n(Y_n - MX^*_{n,n-1}) \\
# H_n &= S^*_{n,n-1}M^\mathsf{T}[R_n + MS^*_{n,n-1}M^\mathsf{T}]^{-1} \\
# S^*_{n,n-1} &= \Phi S^*_{n-1,n-1}\Phi^\mathsf{T} + Q_n \\
# S^*_{n-1,n-1} &= (I-H_{n-1}M)S^*_{n-1,n-2}
# \end{aligned}$$
#
# ## Gelb
#
# $$
# \begin{aligned}
# \underline{\hat{x}}_k(-) &= \Phi_{k-1} \underline{\hat{x}}_{k-1}(+) \\
# \underline{\hat{x}}_k(+) &= \underline{\hat{x}}_k(-) +K_k[Z_k - H_k\underline{\hat{x}}_k(-)] \\
# K_k &= P_k(-)H_k^\mathsf{T}[H_kP_k(-)H_k^\mathsf{T} + R_k]^{-1} \\
# P_k(+) &=  \Phi_{k-1} P_{k-1}(+)\Phi_{k-1}^\mathsf{T} + Q_{k-1} \\
# P_k(-) &= (I-K_kH_k)P_k(-)
# \end{aligned}$$
#
#
# ## Brown
#
# $$
# \begin{aligned}
# \hat{\mathbf x}^-_{k+1} &= \mathbf{\phi}_{k}\hat{\mathbf x}_{k} \\
# \hat{\mathbf x}_k  &= \hat{\mathbf x}^-_k +\mathbf{K}_k[\mathbf{z}_k - \mathbf{H}_k\hat{\mathbf{}x}^-_k] \\
# \mathbf{K}_k &= \mathbf P^-_k\mathbf{H}_k^\mathsf{T}[\mathbf{H}_k\mathbf P^-_k\mathbf{H}_k^T + \mathbf{R}_k]^{-1}\\
# \mathbf P^-_{k+1} &=  \mathbf{\phi}_k \mathbf P_k\mathbf{\phi}_k^\mathsf{T} + \mathbf Q_{k} \\
# \mathbf P_k &= (\mathbf{I}-\mathbf{K}_k\mathbf{H}_k)\mathbf P^-_k
# \end{aligned}$$
#
#
# ## Zarchan
#
# $$
# \begin{aligned}
# \hat{x}_{k} &= \Phi_{k}\hat{x}_{k-1} + G_ku_{k-1} + K_k[z_k - H\Phi_{k}\hat{x}_{k-1} - HG_ku_{k-1} ] \\
# M_{k} &=  \Phi_k P_{k-1}\phi_k^\mathsf{T} + Q_{k} \\
# K_k &= M_kH^\mathsf{T}[HM_kH^\mathsf{T} + R_k]^{-1}\\
# P_k &= (I-K_kH)M_k
# \end{aligned}$$
