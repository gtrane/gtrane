# -*- coding: utf-8 -*-
#
#  Full Model of Glycolysis
#

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def func(y, t, k1, kminus1, k2, kminus2, k3, kminus3, k4, kminus4, k5, kminus5, k6, kminus6, k7, kminus7, k8, kminus8,
         k9, kminus9, k10, kminus10, k11, kminus11, k12, kminus12, k13, kminus13, k14, kminus14,
         k15, kminus15, k16, kminus16, k17, kminus17, k18, kminus18, k19, kminus19, k20, kminus20, k21, kminus21,
         k22, kminus22, k23, kminus23, k24, kminus24, k25, kminus25, k26, kminus26, k27, kminus27, k28, kminus28,
         k29, kminus29, k30, kminus30, k31, kminus31, k32, kminus32, k33, kminus33, k34, kminus34, k35, kminus35,
         k36, kminus36, k37, kminus37, k38, kminus38, k39, kminus39, k40, kminus40, k41, kminus41, k42, kminus42,
         k43, kminus43, k44, kminus44, k45, kminus45, k46, kminus46, k47, kminus47, k48, kminus48, k49, kminus49,
         k50, kminus50, k51, kminus51):
    (hk, glc, hk_glc, mgatp, hk_mgatp_glc, hk_mgatp, g6p_hk_mgadp, hplus, hk_g6p, mgadp, g6p, hk_mgadp,
     pgi, pgi_g6p, f6p, pfk1, pfk1_f6p, f6p_pfk1_mgatp, pfk1_mgatp, f16bp_pfk1_mgadp, pfk1_mgadp, f16bp, pfk1_f16bp,
     aldo, f16bp_aldo, gapd, dhap_aldo, dhap, tpi, dhap_tpi, gapdh, gapd_gapdh, nadplus, gapd_gapdh_nadplus,
     gapdh_nadplus, hpo4, gapd_gapdh_nad_hpo4, bpg13_gapdh_nadh, bpg13, gapdh_nadh, nadh, pgk, mgadp_pgk,
     mgadp_pgk_bpg13, pgk_bpg13, pg3, pgk_mgatp, pgm, pg3_pgm, pg2, eno, pg2_eno, pep_eno, h2o, pep, pk, pk_mgadp,
     pep_pk_mgadp, pk_pep, pyr_pk_mgatp, pyr, pk_mgatp) = y

    dydt = [
        # d(hk)/dt
        -k1 * hk * glc + kminus1 * hk_glc - k3 * hk * mgatp + kminus3 * hk_mgatp + k7 * hk_g6p - kminus7 * hk * g6p
        + k9 * hk_mgadp - kminus9 * hk * mgadp,
        # d(glc)/dt
        -k1 * hk * glc + kminus1 * hk_glc - k4 * glc * hk_mgatp + kminus4 * hk_mgatp_glc,
        # d(hk_glc)/dt
        k1 * hk * glc - kminus1 * hk_glc - k2 * hk_glc * mgatp + kminus2 * hk_mgatp_glc,
        # d(mgatp)/dt
        -k2 * hk_glc * mgatp + kminus2 * hk_mgatp_glc - k3 * hk * mgatp + kminus3 * hk_mgatp
        - k13 * pfk1 * mgatp + kminus13 * pfk1_mgatp - k14 * mgatp * pfk1_f6p + kminus14 * f6p_pfk1_mgatp
        + k39 * pgk_mgatp - kminus39 * pgk * mgatp + k51 * pk_mgatp - kminus51 * pk * mgatp,
        # d(hk_mgatp_glc)/dt
         k2 * hk_glc * mgatp - kminus2 * hk_mgatp_glc + k4 * hk_mgatp * glc - kminus4 * hk_mgatp_glc
        - k5 * hk_mgatp_glc + kminus5 * g6p_hk_mgadp * hplus,
        # d(hk_mgatp)/dt
        k3 * hk * mgatp - kminus3 * hk_mgatp - k4 * hk_mgatp * glc + kminus4 * hk_mgatp_glc,
        # d(g6p_hk_mgadp)/dt
        k5 * hk_mgatp_glc - kminus5 * g6p_hk_mgadp * hplus - k6 * g6p_hk_mgadp + kminus6 * mgadp * hk_g6p
        - k8 * g6p_hk_mgadp + kminus8 * g6p * hk_mgadp,
        # d(hplus)/dt
        k5 * hk_mgatp_glc - kminus5 * g6p_hk_mgadp * hplus + k16 * f6p_pfk1_mgatp
        - kminus16 * f16bp_pfk1_mgadp * hplus + k31 * gapd_gapdh_nad_hpo4 - kminus31 * bpg13_gapdh_nadh * hplus
        - k49 * pep_pk_mgadp * hplus + kminus49 * pyr_pk_mgatp,
        # d(hk_g6p)/dt
        k6 * g6p_hk_mgadp - kminus6 * mgadp * hk_g6p - k7 * hk_g6p + kminus7 * hk * g6p,
        # d(mgadp)/dt
        k6 * g6p_hk_mgadp - kminus6 * hk_g6p * mgadp + k9 * hk_mgadp - kminus9 * hk * mgadp
        + k18 * pfk1_mgadp - kminus18 * pfk1 * mgadp + k19 * f16bp_pfk1_mgadp
        - kminus19 * mgadp * pfk1_f16bp - k34 * mgadp * pgk + kminus34 * mgadp_pgk - k37 * mgadp * pgk_bpg13
        + kminus37 * mgadp_pgk_bpg13 - k45 * pk * mgadp + kminus45 * pk_mgadp - k48 * pk_pep * mgadp
        + kminus48 * pep_pk_mgadp,
        # d(g6p)/dt
        k7 * hk_g6p - kminus7 * hk * g6p + k8 * g6p_hk_mgadp - kminus8 * g6p * hk_mgadp
        - k10 * g6p * pgi + kminus10 * pgi_g6p,
        # d(hk_mgadp)/dt
        k8 * g6p_hk_mgadp - kminus8 * g6p * hk_mgadp - k9 * hk_mgadp + kminus9 * hk * mgadp,
        # d(pgi)/dt
        -k10 * pgi * g6p + kminus10 * pgi_g6p + k11 * pgi_g6p - kminus11 * pgi * f6p,
        # d(pgi_g6p)/dt
        k10 * pgi * g6p - kminus10 * pgi_g6p - k11 * pgi_g6p + kminus11 * pgi * f6p,
        # d(f6p)/dt
        k11 * pgi_g6p - kminus11 * pgi * f6p - k12 * pfk1 * f6p + kminus12 * pfk1_f6p
        - k15 * f6p * pfk1_mgatp + kminus15 * f6p_pfk1_mgatp,
        # d(pfk1)/dt
        -k12 * pfk1 * f6p + kminus12 * pfk1_f6p - k13 * pfk1 * mgatp + kminus13 * pfk1_mgatp + k18 * pfk1_mgadp
        - kminus18 * pfk1 * mgadp + k20 * pfk1_f16bp - kminus20 * pfk1 * f16bp,
        # d(pfk1_f6p)/dt
        k12 * pfk1 * f6p - kminus12 * pfk1_f6p - k14 * mgatp * pfk1_f6p + kminus14 * f6p_pfk1_mgatp,
        # d(f6p_pfk1_mgatp)/dt
        k13 * mgatp * pfk1_f6p - kminus13 * f6p_pfk1_mgatp + k14 * f6p * pfk1_mgatp - kminus14 * f6p_pfk1_mgatp
        - k16 * f6p_pfk1_mgatp + kminus16 * f16bp_pfk1_mgadp * hplus,
        # d(pfk1_mgatp)/dt
        k14 * pfk1 * mgatp - kminus14 * pfk1_mgatp - k13 * f6p * pfk1_mgatp + kminus13 * f6p_pfk1_mgatp,
        # d(f16bp_pfk1_mgadp)/dt
        k16 * f6p_pfk1_mgatp - kminus16 * f16bp_pfk1_mgadp * hplus - k17 * f16bp_pfk1_mgadp
        + kminus17 * f16bp * pfk1_mgadp - k19 * f16bp_pfk1_mgadp + kminus19 * mgadp * pfk1_f16bp,
        # d(pfk1_mgadp)/dt
        k17 * f16bp_pfk1_mgadp - kminus17 * f16bp * pfk1_mgadp - k18 * pfk1_mgadp + kminus18 * pfk1 * mgadp,
        # d(f16bp)/dt
        k17 * f16bp_pfk1_mgadp - kminus17 * f16bp * pfk1_mgadp + k20 * pfk1_f16bp - kminus20 * pfk1 * f16bp
         - k21 * f16bp * aldo + kminus21 * f16bp_aldo,
        # d(pfk1_f16bp)/dt
        k19 * f16bp_pfk1_mgadp - kminus19 * mgadp * pfk1_f16bp - k20 * pfk1_f16bp + kminus20 * pfk1 * f16bp,
        # d(aldo)/dt
        -k21 * f16bp * aldo + kminus21 * f16bp_aldo + k23 * dhap_aldo - kminus23 * dhap * aldo,
        # d(f16bp_aldo)/dt
        k21 * f16bp * aldo - kminus21 * f16bp_aldo - k22 * f16bp_aldo + kminus22 * gapd * dhap_aldo,
        # d(gapd)/dt
        k22 * f16bp_aldo - kminus22 * gapd * dhap_aldo + k25 * dhap_tpi - kminus25 * gapd * tpi - k26 * gapd * gapdh
        + kminus26 * gapd_gapdh - k29 * gapd * gapdh_nadplus + kminus29 * gapd_gapdh_nadplus,
        # d(dhap_aldo)/dt
        k22 * f16bp_aldo - kminus22 * gapd * dhap_aldo - k23 * dhap_aldo + kminus23 * dhap * aldo,
        # d(dhap)/dt
        k23 * dhap_aldo - kminus23 * dhap * aldo - k24 * dhap * tpi + kminus24 * dhap_tpi,
        # d(tpi)/dt
        -k24 * dhap * tpi + kminus24 * dhap_tpi + k25 * dhap_tpi - kminus25 * gapd * tpi,
        # d(dhap_tpi)/dt
        k24 * dhap * tpi - kminus24 * dhap_tpi - k25 * dhap_tpi + kminus25 * gapd * tpi,
        # d(gapdh)/dt
        -k26 * gapd * gapdh + kminus26 * gapd_gapdh - k28 * gapdh * nadplus + kminus28 * gapdh_nadplus
        + k33 * gapdh_nadh - kminus33 * gapdh * nadh,
        # d(gapd_gapdh)/dt
        k26 * gapd * gapdh - kminus26 * gapd_gapdh - k27 * gapd_gapdh * nadplus + kminus27 * gapd_gapdh_nadplus,
        # d(nadplus)/dt
        -k27 * gapd_gapdh * nadplus + kminus27 * gapd_gapdh_nadplus - k28 * gapdh * nadplus + kminus28 * gapdh_nadplus,
        # d(gapd_gapdh_nadplus)/dt
        k27 * gapd_gapdh * nadplus - kminus27 * gapd_gapdh_nadplus + k29 * gapd * gapdh_nadplus
        - kminus29 * gapd_gapdh_nadplus - k30 * gapd_gapdh_nadplus * hpo4 + kminus30 * gapd_gapdh_nad_hpo4,
        # d(gapdh_nadplus/dt)
        k28 * gapdh * nadplus - kminus28 * gapdh_nadplus - k29 * gapd * gapdh_nadplus + kminus29 * gapd_gapdh_nadplus,
        # d(hpo4)/dt
        -k30 * gapd_gapdh_nadplus * hpo4 + kminus30 * gapd_gapdh_nad_hpo4,
        # d(gapd_gapdh_nad_hpo4)/dt
        k30 * gapd_gapdh_nadplus * hpo4 - kminus30 * gapd_gapdh_nad_hpo4 - k31 * gapd_gapdh_nad_hpo4
        + kminus31 * bpg13_gapdh_nadh * hplus,
        # d(bpg13_gapdh_nadh)/dt
        k31 * gapd_gapdh_nad_hpo4 - kminus31 * bpg13_gapdh_nadh * hplus - k32 * bpg13_gapdh_nadh
        + kminus32 * bpg13 * gapdh_nadh,
        # d(bpg13)/dt
        k32 * bpg13_gapdh_nadh - kminus32 * bpg13 * gapdh_nadh - k35 * mgadp_pgk * bpg13 + kminus35 * mgadp_pgk_bpg13
        - k36 * pgk * bpg13 + kminus36 * pgk_bpg13,
        # d(gapdh_nadh)/dt
        k32 * bpg13_gapdh_nadh - kminus32 * bpg13 * gapdh_nadh - k33 * gapdh_nadh + kminus33 * gapdh * nadh,
        # d(nadh)/dt
        k33 * gapdh_nadh - kminus33 * gapdh * nadh,
        # d(pgk)/dt
        -k34 * mgadp * pgk + kminus34 * mgadp_pgk - k36 * pgk * bpg13 + kminus36 * pgk_bpg13 + k39 * pgk_mgatp
        - kminus39 * pgk * mgatp,
        # d(mgadp_pgk)/dt
        k34 * mgadp * pgk - kminus34 * mgadp_pgk - k35 * mgadp_pgk * bpg13 + kminus35 * mgadp_pgk_bpg13,
        # d(mgadp_pgk_bpg13)/dt
        k35 * mgadp_pgk * bpg13 - kminus35 * mgadp_pgk_bpg13 + k37 * pgk_bpg13 * mgadp - kminus37 * mgadp_pgk_bpg13
        - k38 * mgadp_pgk_bpg13 + kminus38 * pg3 * pgk_mgatp,
        # d(pgk_bpg13)/dt
        k36 * pgk * bpg13 - kminus36 * pgk_bpg13 - k37 * pgk_bpg13 * mgadp + kminus37 * mgadp_pgk_bpg13,
        # d(pg3)/dt
        k38 * mgadp_pgk_bpg13 - kminus38 * pg3 * pgk_mgatp - k40 * pg3 * pgm + kminus40 * pg3_pgm,
        # d(pgk_mgatp)/dt
        k38 * mgadp_pgk_bpg13 - kminus38 * pg3 * pgk_mgatp - k39 * pgk_mgatp + kminus39 * pgk * mgatp,
        # d(pgm)/dt
        -k40 * pg3 * pgm + kminus40 * pg3_pgm + k41 * pg3_pgm - kminus41 * pg2 * pgm,
        # d(pg3_pgm)/dt
        k40 * pg3 * pgm - kminus40 * pg3_pgm - k41 * pg3_pgm + kminus41 * pg2 * pgm,
        # d(pg2)/dt
        k41 * pg3_pgm - kminus41 * pg2 * pgm - k42 * pg2 * eno + kminus42 * pg2_eno,
        # d(eno)/dt
        -k42 * pg2 * eno + kminus42 * pg2_eno + k44 * pep_eno - kminus44 * pep * eno,
        # d(pg2_eno)/dt
        k42 * pg2 * eno - kminus42 * pg2_eno - k43 * pg2_eno + kminus43 * pep_eno * h2o,
        # d(pep_eno)/dt
        k43 * pg2_eno - kminus43 * pep_eno * h2o - k44 * pep_eno + kminus44 * pep * eno,
        # d(h2o)/dt
        k43 * pg2_eno - kminus43 * pep_eno * h2o,
        # d(pep)/dt
        k44 * pep_eno - kminus44 * pep * eno - k46 * pep * pk_mgadp + kminus46 * pep_pk_mgadp - k47 * pep * pk
        + kminus47 * pk_pep,
        # d(pk)/dt
        -k45 * pk * mgadp + kminus45 * pk_mgadp - k47 * pk * pep + kminus47 * pk_pep + k51 * pk_mgatp
        - kminus51 * pk * mgatp,
        # d(pk_mgadp)/dt
        k45 * pk * mgadp - kminus45 * pk_mgadp - k46 * pk_mgadp * pep + kminus46 * pep_pk_mgadp,
        # d(pep_pk_mgadp)/dt
        k46 * pk_mgadp * pep - kminus46 * pep_pk_mgadp + k48 * pk_pep * mgadp - kminus48 * pep_pk_mgadp
        - k49 * pep_pk_mgadp * hplus + kminus49 * pyr_pk_mgatp,
        # d(pk_pep)/dt
        k47 * pk * pep - kminus47 * pk_pep - k48 * pk_pep * mgadp + kminus48 * pep_pk_mgadp,
        # d(pyr_pk_mgatp)/dt
        k49 * pep_pk_mgadp * hplus - kminus49 * pyr_pk_mgatp - k50 * pyr_pk_mgatp + kminus50 * pyr * pk_mgatp,
        # d(pyr)/dt
        k50 * pyr_pk_mgatp - kminus50 * pyr * pk_mgatp,
        # d(pk_mgatp)/dt
        k50 * pyr_pk_mgatp - kminus50 * pk_mgatp * pyr - k51 * pk_mgatp + kminus51 * pk * mgatp
        ]
    return dydt


def all_steps(max_time=350):
    """
    Full Model of Glycolysis

    Parameters
    ----------
    max_time: number
        maximum time for integration interval


    Returns
    -------
    graphs describing the results of integrating the model ode's

    """
    #
    # reaction rate constants
    #
    # step 1
    #
    k1      = 1.0  # HK + GLC <=> HK-GLC
    kminus1 = 0.1
    k2      = 1.0  # HK-GLC + MgATP <=> HK-MgATP-GLC
    kminus2 = 0.1
    k3      = 1.0  # HK + MgATP <=> HK-MgATP
    kminus3 = 0.1
    k4      = 1.0  # HK-MgATP + GLC <=> HK-MgATP-GLC
    kminus4 = 0.1
    k5      = 0.1  # HK-MgATP-GLC <=> G6P-HK-MgADP + H+ -- CATALYTIC STEP --
    kminus5 = 0.0001
    k6      = 1.0  # G6P-HK-MgADP <=> MgADP + HK-G6P
    kminus6 = 0.1
    k7      = 1.0  # HK-G6P <=> HK + G6P
    kminus7 = 0.1
    k8      = 1.0  # G6P-HK-MgADP <=> G6P + HK-MgADP
    kminus8 = 0.1
    k9      = 1.0  # HK-MgADP <=> HK + MgADP
    kminus9 = 0.1
    #
    # step 2
    #
    k10     = 1.0  # PGI + G6P <=> PGI-G6P
    kminus10= 0.1
    k11     = 0.1  # PGI-G6P <=> F6P + PGI -- CATALYTIC STEP --
    kminus11= 0.0001
    #
    # step 3
    #
    k12     = 1.0  # PFK1 + F6P <=> PFK1-F6P
    kminus12= 0.1
    k13     = 1.0  # MgATP + PFK1-F6P <=> F6P-PFK1-MgATP
    kminus13= 0.1
    k14     = 1.0  # PFK1 + MgATP <=> PFK1-MgATP
    kminus14= 0.1
    k15     = 1.0  # F6P + PFK1-MgATP <=> F6P-PFK1-MgATP
    kminus15= 0.1
    k16     = 0.1  # F6P-PFK1-MgATP <=> F1,6BP-PFK1-MgADP + H+  -- CATALYTIC STEP --
    kminus16= 0.0001
    k17     = 1.0  # F1,6BP-PFK1-MgADP <=> F1,6BP + PFK1-MgADP
    kminus17= 0.1
    k18     = 1.0  # PFK1-MgADP <=> PFK1 + MgADP
    kminus18= 0.1
    k19     = 1.0  # F1,6BP-PFK1-MgADP <=> MgADP+ PFK1-f16bp
    kminus19= 0.1
    k20     = 1.0  # PFK1-f16bp <=> PFK1 + f16bp
    kminus20= 0.1
    #
    # step 4
    #
    k21     = 1.0  # F16BP + ALDO <=> F16BP-ALDO
    kminus21= 0.1
    k22     = 0.1  # F16BP-ALDO <=> GAPD + DHAP-ALDO -- CATALYTIC STEP --
    kminus22= 0.0001
    k23     = 1.0  # DHAP-ALDO <=> DHAP + ALDO
    kminus23= 0.1
    #
    # step 5
    #
    k24     = 1.0  # DHAP + TPI <=> DHAP-TPI
    kminus24= 0.1
    k25     = 0.1  # DHAP-TPI <=> GAPD + TPI -- CATALYTIC STEP --
    kminus25= 0.0001
    #
    # step 6
    #
    k26     = 1.0  # GAPD + GAPDH <=> GAPD-GAPDH
    kminus26= 0.1
    k27     = 1.0  # GAPD-GAPDH + NADPLUS <=> GAPD-GAPDH-NADPLUS
    kminus27= 0.1
    k28     = 1.0  # GAPDH + NADPLUS <=> GAPDH-NADPLUS
    kminus28= 0.1
    k29     = 1.0  # GAPD + GAPDH-NADPLUS <=> GAPD-GAPDH-NADPLUS
    kminus29= 0.1
    k30     = 1.0  # GAPD-GAPDH-NADPLUS + HPO4 <=> GAPD-GAPDH-NAD-HPO4
    kminus30= 0.1
    k31     = 0.1  # GAPD-GAPDH-NAD-HPO4 <=> BPG13-GAPDH-NADH + HPLUS  -- CATALYTIC STEP --
    kminus31= 0.0001
    k32     = 1.0  # BPG13-GAPDH-NADH <=> BPG13 + GAPDH-NADH
    kminus32= 0.1
    k33     = 1.0  # GAPDH-NADH <=> GAPDH + NADH
    kminus33= 0.1
    #
    # step 7
    #
    k34 = 1.0  # MgADP + PGK <=> MgADP-PGK
    kminus34 = 0.1
    k35 = 1.0  # MgADP-PGK + BPG13 <=> MgADP-PGK-BPG13
    kminus35 = 0.1
    k36 = 1. # PGK + BPG13 <=> PGK-BPG13
    kminus36 = 0.1
    k37 = 1.0  # PGK-BPG13 + MgADP <=> MgADP-PGK-BPG13
    kminus37 = 0.1
    k38 = 0.1  # MgADP-PGK-BPG13 <=> PG3 + PGK-MgATP  -- CATALYTIC STEP --
    kminus38 = 0.0001
    k39 = 1.0  # PGK-MgATP <=> PGK + MgATP
    kminus39 = 0.1
    #
    # step 8
    #
    k40 = 1.0  # PG3 + PGM  <=> PG3-PGM
    kminus40 = 0.1
    k41 = 0.1  # PG3-PGM <=> PG2 + PGM  -- CATALYTIC STEP --
    kminus41 = 0.0001
    #
    # step 9
    #
    k42 = 1.0  # 2PG + ENO <=> 2PG-ENO
    kminus42 = 0.1
    k43 = 0.1  # 2PG-ENO <=> PEP-ENO + H2O  -- CATALYTIC STEP --
    kminus43 = 0.0001
    k44 = 1.0  # PEP-ENO <=> PEP + ENO
    kminus44 = 0.1
    #
    # step 10
    #
    k45 = 1.0  # PK + MgADP <=> PK-MgADP
    kminus45 = 0.1
    k46 = 1.0  # PK-MgADP + PEP <=> PEP-PK-MgADP
    kminus46 = 0.1
    k47 = 1.0  # PK + PEP <=> PK-PEP
    kminus47 = 0.1
    k48 = 1.0  # PK-PEP + MgADP <=> PEP-PK-MgADP
    kminus48 = 0.1
    k49 = 0.1  # PEP-PK-MgADP + H+ <=> PYR-PK-MgATP  -- CATALYTIC STEP --
    kminus49 = 0.0001
    k50 = 1.0  # PYR-PK-MgATP <=> PYR + PK-MgATP
    kminus50 = 0.1
    k51 = 1.0  # PK-MgATP <=> PK + MgATP
    kminus51 = 0.1

    #
    # initial values for concentration variables
    #
    y0 = [
        1.0,    # 0  - hk
        10.0,   # 1  - glc
        0.0,    # 2  - hk_glc
        24.0,    # 3  - mgatp
        0.0,    # 4  - hk_mgatp_glc
        0.0,    # 5  - hk_mgatp
        0.0,    # 6  - g6p_hk_mgadp
        0.0,    # 7  - hplus
        0.0,    # 8  - hk_g6p
        24.0,   # 9  - mgadp
        0.0,    # 10 - g6p
        0.0,    # 11 - hk_mgadp
        1.0,    # 12 - pgi
        0.0,    # 13 - pgi_g6p
        0.0,    # 14 - f6p
        1.0,    # 15 - pfk1
        0.0,    # 16 - pfk1_f6p
        0.0,    # 17 - f6p_pfk1_mgatp
        0.0,    # 18 - pfk1_mgatp
        0.0,    # 19 - f16bp_pfk1_mgadp
        0.0,    # 20 - pfk1_mgadp
        0.0,    # 21 - f16bp
        0.0,    # 22 - pfk1_f16bp
        1.0,    # 23 - aldo
        0.0,    # 24 - f16bp_aldo
        0.0,    # 25 - gapd
        0.0,    # 26 - dhap_aldo
        0.0,    # 27 - dhap
        1.0,    # 28 - tpi
        0.0,    # 29 - dhap_tpi
        2.0,    # 30 - gapdh
        0.0,    # 31 - gapd_gapdh
        24.0,   # 32 - nadplus
        0.0,    # 33 - gapd_gapdh_nadplus
        0.0,    # 34 - gapdh_nadplus
        24.0,   # 35 - hpo4
        0.0,    # 36 - gapd_gapdh_nadplus_hp04
        0.0,    # 37 - bpg13_gapdh_nadh
        0.0,    # 38 - bpg13
        0.0,    # 39 - gapdh_nadh
        0.0,    # 40 - nadh
        2.0,    # 41 - pgk
        0.0,    # 42 - mgadp_pgk
        0.0,    # 43 - mgadp_pgk_bpg13
        0.0,    # 44 - pgk_bpg13
        0.0,    # 45 - pg3
        0.0,    # 46 - pgk_mgatp
        2.0,    # 47 - pgm
        0.0,    # 48 - pg3-pgm
        0.0,    # 49 - pg2
        2.0,    # 50 - eno
        0.0,    # 51 - pg2_eno
        0.0,    # 52 - pep_eno
        0.0,    # 53 - h2o
        0.0,    # 54 - pep
        2.0,    # 55 - pk
        0.0,    # 56 - pk_mgadp
        0.0,    # 57 - pep_pk_mgadp
        0.0,    # 58 - pk_pep
        0.0,    # 59 - pyr_pk_mgatp
        0.0,    # 60 - pyr
        0.0     # 61 - pk_mgatp
    ]

    tspan = np.linspace(0, max_time, 1000)
    sol = odeint(func, y0, tspan, args=(k1, kminus1, k2, kminus2, k3, kminus3, k4, kminus4, k5, kminus5, k6, kminus6,
                                        k7, kminus7, k8, kminus8, k9, kminus9, k10, kminus10, k11, kminus11,
                                        k12, kminus12, k13, kminus13, k14, kminus14, k15, kminus15,
                                        k16, kminus16, k17, kminus17, k18, kminus18, k19, kminus19, k20, kminus20,
                                        k21, kminus21, k22, kminus22, k23, kminus23, k24, kminus24, k25, kminus25,
                                        k26, kminus26, k27, kminus27, k28, kminus28, k29, kminus29, k30, kminus30,
                                        k31, kminus31, k32, kminus32, k33, kminus33, k34, kminus34, k35, kminus35,
                                        k36, kminus36, k37, kminus37, k38, kminus38, k39, kminus39, k40, kminus40,
                                        k41, kminus41, k42, kminus42, k43, kminus43, k44, kminus44, k45, kminus45,
                                        k46, kminus46, k47, kminus47, k48, kminus48, k49, kminus49, k50, kminus50,
                                        k51, kminus51))
    plt.figure(1)
    plt.title('Free Enzymes')
    plt.plot(tspan, sol[:, 0], tspan, sol[:, 12], tspan, sol[:, 15], tspan, sol[:, 23], tspan, sol[:, 28],
             tspan, sol[:, 30],  tspan, sol[:, 41],  tspan, sol[:, 47], tspan, sol[:, 50],  tspan, sol[:, 55])
    plt.xlabel('t [s]')
    plt.ylabel('Concentration')
    plt.legend(['HK', 'PGI', 'PFK1', 'ALDO', 'TPI', 'GAPDH', 'PGK', 'PGM', 'ENO', 'PK'])

    plt.figure(2)
    plt.title('Enzyme-Substrate Complexes #1')
    plt.plot(tspan, sol[:, 2], tspan, sol[:, 8], tspan, sol[:, 13], tspan, sol[:, 16], tspan, sol[:, 22],
             tspan, sol[:, 24], tspan, sol[:, 26], tspan, sol[:, 29])
    plt.xlabel('t [s]')
    plt.ylabel('Concentration')
    plt.legend(['HK-GLC', 'HK-G6P', 'PGI-G6P', 'PFK1_F6P', 'PFK1-F16BP', 'F16BP-ALDO', 'DHAP-ALDO', 'DHAP-TPI'])

    plt.figure(3)
    plt.title('Enzyme-Substrate Complexes #2')
    plt.plot(tspan, sol[:, 31], tspan, sol[:, 44], tspan, sol[:, 48], tspan, sol[:, 51], tspan, sol[:, 52],
            tspan, sol[:, 58])
    plt.xlabel('t [s]')
    plt.ylabel('Concentration')
    plt.legend(['GAPD_GAPDH', 'PGK-1,3BPG', '3PG-PGM', '2PG-ENO', 'PEP-ENO', 'PEP-PK'])

    plt.figure(4)
    plt.title('Enzyme-Energy Carrier Complexes')
    plt.plot(tspan, sol[:, 5], tspan, sol[:, 11], tspan, sol[:, 18], tspan, sol[:, 20], tspan, sol[:, 34],
            tspan, sol[:, 39], tspan, sol[:, 42], tspan, sol[:, 46], tspan, sol[:, 56], tspan, sol[:, 61])
    plt.xlabel('t [s]')
    plt.ylabel('Concentration')
    plt.legend(['HK-MgATP', 'HK-MgADP', 'PFK1-MgATP', 'PFK1-MgADP', 'GAPDH-NADPLUS', 'GAPDH-NADH', 'PGK-MgADP',
                'PGK-MgATP', 'PK-MgADP', 'PK-MgATP'])

    plt.figure(5)
    plt.title('Ternary Complexes')
    plt.plot(tspan, sol[:, 4], tspan, sol[:, 6], tspan, sol[:, 17], tspan, sol[:, 19], tspan, sol[:, 33],
            tspan, sol[:, 37], tspan, sol[:, 43], tspan, sol[:, 57], tspan, sol[:, 59])
    plt.xlabel('t [s]')
    plt.ylabel('Concentration')
    plt.legend(['HK-MgATP-GLC', 'G6P-HK-MgADP', 'F6P-PFK1-MgATP', 'F16BP-PFK1-MgADP', 'GAPD-GAPDH-NADPLUS',
                '1,3BPG-GAPDH-NADH', '1,3BPG-PGK-MgADP', 'PEP-PK-MgADP', 'PYR-PK-MgATP'])

    plt.figure(6)
    plt.title('Quaternary Complexes')
    plt.plot(tspan, sol[:, 36])
    plt.xlabel('t [s]')
    plt.ylabel('Concentration')
    plt.legend(['GAPD-GAPDH-NADPLUS-HPO4'])

    plt.figure(7)
    plt.title('Energy Carriers and Metabolites')
    plt.plot(tspan, sol[:, 3], tspan, sol[:, 9], tspan, sol[:, 32], tspan, sol[:, 40], tspan, sol[:, 7],
             tspan, sol[:, 35], tspan, sol[:, 53])
    plt.xlabel('t [s]')
    plt.ylabel('Concentration')
    plt.legend(['MgATP', 'MgADP', 'NAD+', 'NADH', 'H+', 'HPO4', 'H20'])

    plt.figure(8)
    plt.title('Substrates and Product')
    plt.plot(tspan, sol[:, 1], tspan, sol[:, 10], tspan, sol[:, 14], tspan, sol[:, 21], tspan, sol[:, 25],
             tspan, sol[:, 27], tspan, sol[:, 38], tspan, sol[:, 45],  tspan, sol[:, 49], tspan, sol[:, 54],
             tspan, sol[:, 60])
    plt.xlabel('t [s]')
    plt.ylabel('Concentration')
    plt.legend(['Glucose', 'Glucose-6-phosphate', 'Fructose-6-Phosphate', 'Fructose-1,6-Bisphosphate', 'GAPD', 'DHAP',
                '1,3BPG', '3PG', '2PG', 'PEP', 'Pyruvate'])

    plt.show()


all_steps()

