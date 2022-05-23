import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from copy import copy


# function definition for creating a table
def print_array_pretty(inarray):
    colwidth = 10
    for i, row in enumerate(inarray):
        outrow = ''
        for j, item in enumerate(row):
            if j == 0:
                outrow += inarray[i, j].ljust(colwidth)
            else:
                if i == 0:
                    outrow += ' ' * (colwidth - 7) + inarray[i, j].ljust(7)
                else:
                    outrow += format(float(inarray[i, j]), '.5f').rjust(colwidth)
        print(outrow)


# 1/3-octave bands
Fref = np.array(
    [0.8, 1, 1.25, 1.6, 2, 2.5, 3.15, 4, 5, 6.3, 8, 10, 12.5, 16, 20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160,
     200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500,
     16000, 20000]).reshape(-1, 1)

# 4.2 Loading data from Appendix E
F = np.array([96.9, 99.6, 102.3, 105, 107.7, 110.4, 113, 115.7, 118.4, 121.1, 123.8,
              126.5, 129.2, 131.9, 134.6, 137.3, 140, 142.7, 145.3, 148, 150.7, 153.4,
              156.1, 158.8, 161.5, 164.2, 166.9, 169.6, 172.3, 175, 177.6, 180.3, 183,
              185.7, 188.4, 191.1, 193.8, 196.5]).reshape(-1, 1)
L_i = np.array([49.4, 50.68, 50.09, 53.37, 44.47, 50.91, 51.41, 59.40,
                64.54, 57.57, 51.02, 50.76, 59.93, 62.94, 58.49, 65.87, 62.66, 50.25,
                51.32, 52.30, 52.58, 53.15, 67.04, 67.27, 57.40, 57.17, 52.56, 51.39,
                52.49, 47.68, 51.26, 49.03, 61.42, 59.52, 48.43, 50.84, 48.20, 55.95]).reshape(-1, 1)
pref = 20e-6
deltaf = 2.7
deltaf_e = 1.5 * deltaf

# # plot check
# figure_size = (7, 5)
# x_tick = np.log10(Fref)
# plt.figure(figsize=figure_size)
# plt.title("Plot data", fontsize=12)
# plt.xlabel("Frequency, Hz", fontsize=12)
# plt.ylabel("SPL, dB", fontsize=12)
# # plt.grid()
# # plt.semilogx(F, L_i, "o", mec='k', mew=0.5)
# plt.semilogx(F, L_i)
# ax = plt.gca()
# ax.xaxis.set_minor_formatter(mticker.ScalarFormatter())
# plt.xticks(x_tick)
# plt.xlim([95, 200])
# plt.show()

# Calculate critical bands
CB = np.empty(shape=(len(F), 1), dtype=float)
CB_edges = np.empty(shape=(len(F), 2), dtype=float)
CB_lines = np.empty(shape=(len(F), 2), dtype=float)
a_v = np.empty(shape=(len(F), 1), dtype=float)

for ii in range(len(F)):
    a_v[ii] = -2 - np.log10(1 + (F[ii] / 502) ** 2.5)  # masking index
    CB[ii] = 25 + 75 * (1 + 1.4 * (F[ii] / 1000) ** 2) ** 0.69  # eq(2)
    CB_edges[ii, 0] = -0.5 * CB[ii] + ((CB[ii] ** 2 + 4 * F[ii] ** 2) ** 0.5) / 2  # eq(4)
    CB_edges[ii, 1] = min((CB_edges[ii, 0] + CB[ii]), F[-1])  # eq(5)
    Ftemp1 = copy(F)
    Ftemp2 = copy(F)
    Ftemp1[F < CB_edges[ii, 0]] = float("nan")
    Ftemp2[F > CB_edges[ii, 1]] = float("nan")
    arr_temp1 = Ftemp1 - CB_edges[ii, 0]
    min_p1 = np.nanmin(arr_temp1)
    f1_idx = np.nanargmin(arr_temp1)
    arr_temp2 = CB_edges[ii, 1] - Ftemp2
    min_p2 = np.nanmin(arr_temp2)
    f2_idx = np.nanargmin(arr_temp2)
    CB_lines[ii, 0] = f1_idx
    CB_lines[ii, 1] = f2_idx

# 5.3.2 Determination of the mean narrow-band level L_S of the masking noise (only necessary when there is a tone)
# This procedure also corresponds to Detailed Diagram 1

L_S = np.empty(shape=(len(F), 1), dtype=float)
L_Stemp = copy(L_S)
Ftemp = copy(F)

for ii in range(len(F)):
    idx_Tone = np.array(range(int(CB_lines[ii, 0]), int(CB_lines[ii, 1]) + 1))
    L_temp = L_i[idx_Tone]
    index_fcenter = int(np.where(idx_Tone == ii)[0])
    L_temp[index_fcenter] = float("nan")
    idx_preTone = 4
    idx_postTone = 4
    L_Stemp1 = 0
    deltaL = 1

    while deltaL > 0.005 and idx_preTone >= 4 and idx_postTone >= 4:
        L_Stemp = 10 * np.log10(np.nanmean(10 ** (L_temp / 10))) + 10 * np.log10(deltaf / deltaf_e)
        L_temp[L_temp > (L_Stemp + 6)] = float("nan")
        deltaL = abs(L_Stemp1 - L_Stemp)
        L_Stemp1 = copy(L_Stemp)
        idx_end = index_fcenter - 1
        if idx_end < 0:
            idx_end = 0
        idx_preTone = len(L_temp[0:idx_end]) - sum(pd.isna(L_temp[0:idx_end]))
        idx_postTone = len(L_temp[index_fcenter + 1:-1]) - sum(pd.isna(L_temp[index_fcenter + 1: -1]))

    L_S[ii] = copy(L_Stemp)

# 5.3.3 Determination of the tone level LT of a tone in a critical band
Tone_Index = np.zeros((len(F), 1))
for ii in range(1, len(F)):
    # print(ii)
    if L_i[ii] > L_S[ii] + 6:
        if L_i[ii] > L_i[ii - 1] and L_i[ii] > L_i[ii + 1]:  # ensures that the largest peak is chosen for 3 adjacent
            # frequency bins (for each spectrum)
            Tone_Index[ii] = 1
        else:
            Tone_Index[ii] = 0
    else:
        Tone_Index[ii] = 0

f_index = np.where(Tone_Index != 0)[0]
F_tonA = F[f_index]  # finding the associated frequency of the maximum tonal peak
L_T = np.empty(shape=(len(F), 1), dtype=float)
L_T[:] = 0
L_T[f_index] = L_i[f_index]  # we consider all tones here, not just the maxima in the CB
L_T1 = copy(L_T)

# Combining adjacent lines
idx_all_T = np.empty(shape=(100, len(F_tonA)), dtype=float) * 0

for ii in range(len(F_tonA)):  # considering the first tone and then subsequent tones
    # print(ii)
    kk = 1
    L_T_idx = L_T[f_index[ii]]  # tone level
    L_i_idx = L_i[f_index[ii] + kk]  # first spectral line above the tone
    L_S_idx = L_S[f_index[ii]]  # mean narrow-band level
    Fu = F[f_index[ii] - kk]
    Lu = L_i[f_index[ii] - kk]  # first spectral line below the tone
    Fo = F[f_index[ii] + kk]
    Lo = copy(L_i_idx)
    cond1 = False
    cond2 = False

    # breakpoint()
    while L_i_idx + 10 > L_T_idx and L_i_idx > L_S_idx + 6 and f_index[ii] + kk <= len(L_i):
        kk = kk + 1
        L_i_idx_temp = copy(L_i_idx)
        L_i_idx = L_i[f_index[ii] + kk]
        if L_i_idx > L_T_idx:
            cond1 = True
            break

    idx_break1 = copy(kk) - 1
    kk = 1
    L_i_idx = L_i[f_index[ii] - kk]

    while L_i_idx + 10 > L_T_idx and L_i_idx > L_S_idx + 6 and f_index[ii] > kk + 1:  # last condition ensures that
        # we don't go beyond the length of L_i
        kk = kk + 1
        L_i_idx_temp = copy(L_i_idx)
        L_i_idx = L_i[f_index[ii] - kk]

        if L_i_idx > L_T_idx:
            cond2 = True
            break

    idx_break2 = copy(kk) - 1
    # breakpoint()
    if cond1 or cond2:  # yes, not highest peaks
        L_T[f_index[ii]] = 0
    else:
        idx_all = np.array(range(int(f_index[ii] - idx_break2), int(idx_break1 + f_index[ii]) + 1))
        idx_all_T[np.arange(0, len(idx_all)), ii] = idx_all
        if len(idx_all) == 1:
            L_T_temp = L_T_idx  # one tone
        else:
            L_T_temp = 10 * np.log10(sum(10 ** (L_i[idx_all] / 10))) + 10 * np.log10(deltaf / deltaf_e) \
                # multiple tones

        # 5.3.4 Distinctness of a tone
        if idx_break2 == 0:
            Lu_new = Lu
            Fu_new = Fu
        else:
            Lu_new = L_i[f_index[ii] - idx_break2]  # this is up to interpretation but basically excludes all
            # lines considered as a "tone"
            Fu_new = F[f_index[ii] - idx_break2]

        if idx_break1 == 0:
            Lo_new = Lo
            Fo_new = Fo
        else:
            Lo_new = L_i[idx_break1 + f_index[ii]]
            Fo_new = F[idx_break1 + f_index[ii]]

        delta_Lu = (F_tonA[ii] / 2) * ((L_T_idx - Lu_new) / (F_tonA[ii] - Fu))
        delta_Lo = (F_tonA[ii]) * ((L_T_idx - Lo_new) / (Fo - F_tonA[ii]))
        delta_FR = 26 * (1 + 1e-3 * F_tonA[ii])
        count = idx_break1 + idx_break2 + 1
        BW = count * deltaf
        if delta_Lu >= 24 and delta_Lo >= 24 and BW <= delta_FR:
            L_T[f_index[ii]] = L_T_temp
        else:
            L_T[f_index[ii]] = L_T_temp
            Ftemp[f_index[ii]] = float("nan")

L_G = L_S + 10 * np.log10(CB / deltaf)  # EQ (12)
deltaL = L_T - L_G - a_v
audibleL = deltaL > 0
F_nan = ~np.isnan(Ftemp)
audibleL_sharp = np.logical_and(audibleL, F_nan)

# 5.3.8 Determination of decisive audibility (Step 3 only)

f1 = CB_edges[:, 0].reshape(-1, 1)
f2 = CB_edges[:, 1].reshape(-1, 1)

L_Tm = copy(F) * 0
deltaL_m = copy(F) * 0
reportF = copy(F) * 0
idx11 = copy(F) * 0
idx22 = copy(F) * 0
U = copy(F) * 0
U_m = copy(F) * 0
datanew = np.concatenate((F, L_S, L_T, L_G, a_v, deltaL, f1, f2, CB, L_Tm, deltaL_m, reportF, idx11, idx22, U, U_m), axis=1)
idx_audible = np.where(audibleL_sharp == True)[0]
idx_audible_all = np.where(audibleL == True)[0]

if sum(audibleL_sharp) > 0:
    data_audible = datanew[idx_audible][:]  # including sharp tones only for reporting
    data_audible_all = datanew[idx_audible_all][:]
    inter = np.intersect1d(F_tonA, F[audibleL])
    Fton_all_idx0 = np.argwhere(np.in1d(F_tonA, F[audibleL]))

    if len(data_audible) != 0:
        for ii in range(len(data_audible)):
            idx1 = np.argwhere((np.logical_and(data_audible_all[:, 0] >= data_audible_all[ii, 6],
                                               data_audible_all[:, 0] <= data_audible_all[ii, 7])))
            data_audible[ii, 9] = 10 * np.log10(sum(10. ** (data_audible_all[idx1, 2] / 10)))  # sum audible tone in CB
            idx2 = np.argwhere(
                (np.logical_and(data_audible[:, 0] >= data_audible[ii, 6], data_audible[:, 0] <= data_audible[ii, 7])))
            # find all audible tones in critical band
            data_audible[ii, 10] = data_audible[ii, 9] - data_audible[ii, 3] - data_audible[ii, 4]  # tonal audibility
            maxdL = np.argmax(data_audible[idx2, 5])
            data_audible[ii, 11] = data_audible[maxdL, 0]

        if len(data_audible_all) != 0:
            for ii in range(len(data_audible_all)):
                idx_start = np.argwhere(idx_all_T[:, Fton_all_idx0[ii, 0]] != 0)[0]
                idx_end = np.argwhere(idx_all_T[:, Fton_all_idx0[ii, 0]] != 0)[-1]
                data_audible_all[ii, 12] = idx_all_T[idx_start, Fton_all_idx0[ii]]
                data_audible_all[ii, 13] = idx_all_T[idx_end, Fton_all_idx0[ii]]

        maxdL_m = np.argmax((data_audible[:, 10]))
        data_out = data_audible[maxdL_m, [0, 10, 6, 7, 1, 9, 3, 4]]
        data_out_1tone = data_audible[maxdL_m, [0, 5, 6, 7, 1, 2, 3, 4]]
    else:
        reportF = 0
        L_S = 0
        L_G = 0
        L_Tm = 0
        a_v = 0
        deltaLm = -10
        f1 = 0
        f2 = 0
        data_out = np.array([reportF, L_S, L_Tm, L_G, a_v, deltaLm])
        data_out_1tone = np.array([reportF, L_S, L_Tm, L_G, a_v, deltaLm])
else:
    reportF = 0
    L_S = 0
    L_G = 0
    L_Tm = 0
    a_v = 0
    deltaLm = -10
    f1 = 0
    f2 = 0
    data_out = np.array([reportF, L_S, L_Tm, L_G, a_v, deltaLm])
    data_out_1tone = np.array([reportF, L_S, L_Tm, L_G, a_v, deltaLm])

# Uncertainty calculation
if 'data_audible_all' in dir():
    Fton_idx = np.argwhere(F == data_out[0])[0, 0]
    f_index_ton = np.argwhere(data_audible_all[:, 0] == data_out[0])

    all_tone_indices = []
    # find indices of all tones in critical band
    for ii in range(len(data_audible_all)):
        all_tone_indices = np.append(all_tone_indices, np.array(range(int(data_audible_all[ii, 12]), int(data_audible_all[ii, 13] + 1))))
    
    all_tone_indices = np.array(list(map(int, all_tone_indices)))

    # find indices for tone in question only
    main_tone_index = np.array(range(int(data_audible_all[f_index_ton, 12]), int(data_audible_all[f_index_ton, 13] + 1)))

    idx_back = np.setdiff1d(np.array(range(0, 38)), all_tone_indices)
    LS_secondary = L_i[idx_back]
    FS_secondary = F[idx_back]

    LT_secondary = L_i[main_tone_index]  # only tones around 137.3 Hz
    F_secondary = F[main_tone_index]

    LT_secondary_all = L_i[all_tone_indices]  # all lines classified as a tone in the critical band
    F_secondary_all = F[all_tone_indices]

    # eq (27)
    arg1 = sum((10 ** (LT_secondary / 10)) ** 2)
    arg2 = sum(10 ** (LT_secondary / 10)) ** 2
    arg1_all = sum((10 ** (LT_secondary_all / 10)) ** 2)
    arg2_all = sum(10 ** (LT_secondary_all / 10)) ** 2
    arg3 = sum((10 ** (LS_secondary / 10)) ** 2)
    arg4 = sum(10 ** (LS_secondary / 10)) ** 2
    sigmaL = 3

    sigma_delL = ((arg1 / arg2 + arg3 / arg4) * sigmaL ** 2 + ((4.34 * (deltaf / CB[Fton_idx])) ** 2)) ** 0.5
    U = sigma_delL * 1.645
    data_out_1tone = np.append(data_out_1tone, U)

    sigma_delL_all = ((arg1_all / arg2_all + arg3 / arg4) * sigmaL ** 2 + ((4.34 * (deltaf / CB[Fton_idx])) ** 2)) ** 0.5
    U_all = sigma_delL_all * 1.645
    data_out = np.append(data_out, U_all)

# displaying data in tables
data_list_1tone = ["F", "deltaL", "f_1", "f_2", "L_S", "L_T", "L_G", "a_v", "U"]
data_table_1tone = np.vstack([data_list_1tone, data_out_1tone])
print_array_pretty(data_table_1tone)

print('\n')

data_list = ["F", "deltaL_m", "f_1", "f_2", "L_S", "L_Tm", "L_G", "a_v", "U_m"]
data_table = np.vstack([data_list, data_out])
print_array_pretty(data_table)
