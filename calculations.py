# calculations.py

##########################
### packages to import ###
##########################

import pandas as pd
from datetime import date
import gmpy2 as gp
import melt_gas as mg
import equilibrium_equations as eq


###########################
### saturation pressure ###
###########################

# for a given melt composition, calcualte the saturation pressure
def P_sat(run,PT,melt_wf,setup,species,models,Ptol,nr_step,nr_tol):
    ST = melt_wf["ST"]
    melt_wf1 = melt_wf # to work out P_sat
    melt_wf2 = melt_wf # to work out sulphur saturation
    
    def Pdiff(guess,run,melt_wf,setup,species,models):
        PT["P"] = guess
        difference = abs(guess - mg.p_tot(run,PT,melt_wf,setup,species,models))
        return difference

    guess0 = 40000. # initial guess for pressure
    PT["P"] = guess0
    melt_wf1["Fe3FeT"] = mg.Fe3FeT_i(run,PT,setup,species,models)
    melt_wf2["Fe3FeT"] = mg.Fe3FeT_i(run,PT,setup,species,models)
    wm_S2m_, wm_S6p_, S6p_ST, S2m_ST = eq.melt_speciation(run,PT,melt_wf1,setup,species,models,nr_step,nr_tol)
    melt_wf1["S2-"] = wm_S2m_
    melt_wf2["S2-"] = wm_S2m_
    melt_wf1["ST"] = wm_S2m_+wm_S6p_
    melt_wf2["ST"] = ST
    if models.loc["sulphur_saturation","option"] == "yes":
        SCSS_,sulphide_sat,SCAS_,sulphate_sat, ss_ST = sulphur_saturation(run,PT,melt_wf2,setup,species,models)
        melt_wf1["ST"] = ss_ST/1000000.  
    delta1 = Pdiff(guess0,run,melt_wf1,setup,species,models)
    while delta1 > Ptol :
        delta1 = Pdiff(guess0,run,melt_wf1,setup,species,models)
        guess0 = mg.p_tot(run,PT,melt_wf1,setup,species,models)
        guess0 = float(guess0)
        PT["P"] = guess0
        melt_wf1["Fe3FeT"] = mg.Fe3FeT_i(run,PT,setup,species,models)
        melt_wf2["Fe3FeT"] = mg.Fe3FeT_i(run,PT,setup,species,models)
        wm_S2m_, wm_S6p_, S6p_ST, S2m_ST = eq.melt_speciation(run,PT,melt_wf1,setup,species,models,nr_step,nr_tol)            
        melt_wf1["S2-"] = wm_S2m_
        melt_wf2["S2-"] = wm_S2m_
        melt_wf1["ST"] = wm_S2m_+wm_S6p_
        melt_wf2["ST"] = ST
        if models.loc["sulphur_saturation","option"] == "yes":
            SCSS_,sulphide_sat,SCAS_,sulphate_sat,ss_ST = sulphur_saturation(run,PT,melt_wf2,setup,species,models)
            melt_wf1["ST"] = ss_ST/1000000.
    else:
        P_sat = guess0
        wm_S2m_, wm_S6p_, S6p_ST, S2m_ST = eq.melt_speciation(run,PT,melt_wf1,setup,species,models,nr_step,nr_tol)
        
    melt_wf["ST"] = ST
    return P_sat, wm_S2m_, wm_S6p_, S6p_ST, S2m_ST

# for a given fS2 and fO2, calculate Psat
def P_sat_fO2_fS2(run,PT,melt_wf,setup,species,models,Ptol):
    
    def Pdiff(guess,run,melt_wf,setup,species,models):
        PT["P"] = guess
        P_tot, wm_ST, fSO2, wm_S2, wm_S6, pS2, pO2, pSO2, xgS2, xgO2, xgSO2 = eq.p_tot_fO2_fS2(run,PT,melt_wf,setup,species,models)
        difference = abs(guess - P_tot)
        return difference

    guess0 = 40000. # initial guess for pressure
    PT["P"] = guess0
    delta1 = Pdiff(guess0,run,melt_wf,setup,species,models)
    
    while delta1 > Ptol :
        delta1 = Pdiff(guess0,run,melt_wf,setup,species,models)
        P_tot, wm_ST, fSO2, wm_S2, wm_S6, pS2, pO2, pSO2, xgS2, xgO2, xgSO2 = eq.p_tot_fO2_fS2(run,PT,melt_wf,setup,species,models)
        guess0 = P_tot
        guess0 = float(guess0)
        PT["P"] = guess0
    else:
        P_tot, wm_ST, fSO2, wm_S2, wm_S6, pS2, pO2, pSO2, xgS2, xgO2, xgSO2 = eq.p_tot_fO2_fS2(run,PT,melt_wf,setup,species,models)
    
    return P_tot, wm_ST, fSO2, wm_S2, wm_S6, pS2, pO2, pSO2, xgS2, xgO2, xgSO2

# calculate the saturation pressure for multiple melt compositions and volatile concentrations in input file
def P_sat_output(first_row,last_row,p_tol,nr_step,nr_tol,setup,species,models):
    # set up results table
    results = pd.DataFrame([["oxygen fugacity","sulphide solubility","sulphate solubility","sulphide saturation","ideal gas","Saturation calculation","Date"]])
    results1 = pd.DataFrame([[models.loc["fO2","option"],models.loc["sulphide","option"],models.loc["sulphate","option"],models.loc["sulphur_saturation","option"],models.loc["ideal_gas","option"],models.loc['calc_sat','option'],date.today()]])
    results = results.append(results1, ignore_index=True)
    results1 = ([["Sample","Saturation pressure (bar)","T ('C)","fO2 (DNNO)","fO2 (DFMQ)",
                  "SiO2 (wt%)","TiO2 (wt%)","Al2O3 (wt%)","FeOT (wt%)","MnO (wt%)","MgO (wt%)","CaO (wt%)","Na2O (wt%)","K2O (wt%)","P2O5 (wt%)","ST (ppm)","Fe3/FeT","S2- (ppm)","S6+ (ppm)","S2-/ST", "S6+/ST",
                "SCSS (ppm)","sulphide saturated","SCAS (ppm)","anhydrite saturated","S melt (ppm)",
                "fO2","fS2","fSO2",
                "yO2","yS2","ySO2",
                "pO2","pS2","pSO2",
                "xgO2","xgS2","xgSO2"]])
    results = results.append(results1, ignore_index=True)

    for n in range(first_row,last_row,1): # n is number of rows of data in conditions file
        run = n
        PT={"T":setup.loc[run,"T_C"]}
        if models.loc["calc_sat","option"] == "fO2_fX":
            P_sat_, wm_ST, fSO2, wm_S2m = P_sat_fO2_fS2(run,PT,melt_wf,setup,species,models,p_tol)
            PT["P"] = P_sat_
        else:
            wm_ST = setup.loc[run,"STppm"]/1000000.
        melt_wf = {'ST':wm_ST}
        if setup.loc[run,"S6ST"] > 0.:
            melt_wf["S6ST"] = setup.loc[run,"S6ST"]
        if models.loc["bulk_composition","option"] == "yes":
            bulk_wf = {"S":wm_ST}
        else:
            print("This needs fixing")
        P_sat_, wm_S2m_, wm_S6p_, S6p_ST, S2m_ST = P_sat(run,PT,melt_wf,setup,species,models,p_tol,nr_step,nr_tol)
        PT["P"] = P_sat_
        melt_wf["S2-"] = wm_S2m_
        melt_wf["Fe3FeT"] = mg.Fe3FeT_i(run,PT,setup,species,models)
        SCSS_,sulphide_sat,SCAS_,sulphate_sat, ss_ST = sulphur_saturation(run,PT,melt_wf,setup,species,models)
        gas_mf = {"O2":mg.xg_O2(run,PT,melt_wf,setup,species,models),"S2":mg.xg_S2(run,PT,melt_wf,setup,species,models),"SO2":mg.xg_SO2(run,PT,melt_wf,setup,species,models),"Xg_t":mg.Xg_tot(run,PT,melt_wf,setup,species,models),"wt_g":0.}

        # forward calculate H, C, and S in the melt from reduced species
        wm_S6p_, S2m_ST, S6p_ST, S_T = mg.conc_insolubles(run,PT,melt_wf,setup,species,models)
       
        ### store results ###
        results2 = pd.DataFrame([[setup.loc[run,"Sample"],PT["P"],
                setup.loc[run,"T_C"],mg.fO22Dbuffer(PT,mg.f_O2(run,PT,melt_wf,setup,species,models),"NNO"),mg.fO22Dbuffer(PT,mg.f_O2(run,PT,melt_wf,setup,species,models),"FMQ"),setup.loc[run,"SiO2"],setup.loc[run,"TiO2"],setup.loc[run,"Al2O3"],mg.Wm_FeOT(run,setup,species),setup.loc[run,"MnO"],setup.loc[run,"MgO"],setup.loc[run,"CaO"],setup.loc[run,"Na2O"],setup.loc[run,"K2O"],setup.loc[run,"P2O5"],wm_ST*1000000.,melt_wf["Fe3FeT"],melt_wf['S2-']*1000000.,wm_S6p_*1000000.,S2m_ST, S6p_ST, SCSS_,sulphide_sat,SCAS_,sulphate_sat,ss_ST,
                mg.f_O2(run,PT,melt_wf,setup,species,models),mg.f_S2(run,PT,melt_wf,setup,species,models),mg.f_SO2(run,PT,melt_wf,setup,species,models),
                mg.y_O2(PT,species,models),mg.y_S2(PT,species,models),mg.y_SO2(PT,species,models),
                mg.p_O2(run,PT,melt_wf,setup,species,models),mg.p_S2(run,PT,melt_wf,setup,species,models),mg.p_SO2(run,PT,melt_wf,setup,species,models),       mg.xg_O2(run,PT,melt_wf,setup,species,models),mg.xg_S2(run,PT,melt_wf,setup,species,models),mg.xg_SO2(run,PT,melt_wf,setup,species,models)]])
                             
        results = results.append(results2, ignore_index=True)
        results.to_csv('saturation_pressures.csv', index=False, header=False)
        print(n, setup.loc[run,"Sample"],PT["P"])

# calculate the saturation pressure for multiple melt compositions and fS2 in input file       
def P_sat_output_fS2(first_row,last_row,p_tol,nr_step,nr_tol,setup,species,models):
    # set up results table
    results = pd.DataFrame([["oxygen fugacity","sulphide solubility","sulphate solubility","sulphide saturation","ideal gas","Saturation calculation","Date"]])
    results1 = pd.DataFrame([[models.loc["fO2","option"],models.loc["sulphide","option"],models.loc["sulphate","option"],models.loc["sulphur_saturation","option"],models.loc["ideal_gas","option"],models.loc['calc_sat','option'],date.today()]])
    results = results.append(results1, ignore_index=True)
    results1 = ([["Sample","Pressure (bar)","Saturation pressure (bars)","T ('C)","fO2 (DNNO)","fO2 (DFMQ)",
                  "SiO2 (wt%)","TiO2 (wt%)","Al2O3 (wt%)","FeOT (wt%)","MnO (wt%)","MgO (wt%)","CaO (wt%)","Na2O (wt%)","K2O (wt%)","P2O5 (wt%)","ST (ppm)","S6/ST","Fe3/FeT",
                  "SCSS (ppm)","sulphide saturated","SCAS (ppm)","anhydrite saturated","S melt (ppm)",
                "f-fO2","f-fS2","f-fSO2","f-pO2","f-pS2","f-pSO2","f-xgO2","f-xgS2","f-xgSO2",
                 "b-fO2","b-fS2","b-fSO2","b-pO2","b-pS2","b-pSO2","b-xgO2","b-xgS2","b-xgSO2","ySO2"]])
    results = results.append(results1, ignore_index=True)

    for n in range(first_row,last_row,1): # n is number of rows of data in conditions file
        run = n
        PT={"T":setup.loc[run,"T_C"]}
        wm_ST = setup.loc[run,"STppm"]/1000000.
        melt_wf = {'ST':wm_ST}
        P_sat_, wm_ST, fSO2, wm_S2m, wm_S6p, pS2, pO2, pSO2, xgS2, xgO2, xgSO2 = P_sat_fO2_fS2(run,PT,melt_wf,setup,species,models,p_tol)
        if setup.loc[run,"P_bar"] > 0.:
            PT["P"] = setup.loc[run,"P_bar"]
        else:
            PT["P"] = P_sat_
        melt_wf["ST"] = wm_ST
        melt_wf["S2-"] = wm_S2m
        melt_wf["Fe3FeT"] = mg.Fe3FeT_i(run,PT,setup,species,models)
        SCSS_,sulphide_sat,SCAS_,sulphate_sat, ss_ST = sulphur_saturation(run,PT,melt_wf,setup,species,models)
        gas_mf = {"O2":mg.xg_O2(run,PT,melt_wf,setup,species,models),"S2":mg.xg_S2(run,PT,melt_wf,setup,species,models),"SO2":mg.xg_SO2(run,PT,melt_wf,setup,species,models)}
        
        ### store results ###
        results2 = pd.DataFrame([[setup.loc[run,"Sample"],PT["P"],P_sat_,
                setup.loc[run,"T_C"],mg.fO22Dbuffer(PT,mg.f_O2(run,PT,melt_wf,setup,species,models),"NNO"),mg.fO22Dbuffer(PT,mg.f_O2(run,PT,melt_wf,setup,species,models),"FMQ"),setup.loc[run,"SiO2"],setup.loc[run,"TiO2"],setup.loc[run,"Al2O3"],mg.Wm_FeOT(run,setup,species),setup.loc[run,"MnO"],setup.loc[run,"MgO"],setup.loc[run,"CaO"],setup.loc[run,"Na2O"],setup.loc[run,"K2O"],setup.loc[run,"P2O5"],wm_ST,mg.S6ST(run,PT,melt_wf,setup,species,models),melt_wf["Fe3FeT"],SCSS_,sulphide_sat,SCAS_,sulphate_sat,ss_ST,
                mg.f_O2(run,PT,melt_wf,setup,species,models),setup.loc[run,"fS2"],fSO2, pO2,pS2,pSO2, xgO2,xgS2,xgSO2,mg.f_O2(run,PT,melt_wf,setup,species,models),mg.f_S2(run,PT,melt_wf,setup,species,models),mg.f_SO2(run,PT,melt_wf,setup,species,models), mg.p_O2(run,PT,melt_wf,setup,species,models),mg.p_S2(run,PT,melt_wf,setup,species,models),mg.p_SO2(run,PT,melt_wf,setup,species,models), mg.xg_O2(run,PT,melt_wf,setup,species,models),mg.xg_S2(run,PT,melt_wf,setup,species,models),mg.xg_SO2(run,PT,melt_wf,setup,species,models),mg.y_SO2(PT,species,models)]])
                             
        results = results.append(results2, ignore_index=True)
        results.to_csv('saturation_pressures_fS2.csv', index=False, header=False)
        print(n, setup.loc[run,"Sample"],PT["P"])
        
        
###################
### composition ###
###################

# calculate bulk composition, including if a gas phase is present
def bulk_composition(run,PT,melt_wf,setup,species,models):
    bulk_composition = models.loc["bulk_composition","option"]
    eq_Fe = models.loc["eq_Fe","option"]
    wm_ST = melt_wf["ST"]
    Fe3FeT = melt_wf["Fe3FeT"]
    Fe3Fe2_ = mg.Fe3Fe2(melt_wf)
    S6ST_ = mg.S6ST(run,PT,melt_wf,setup,species,models)
    #SCSS_,sulphide_sat,SCAS_,sulphate_sat = sulphur_saturation(wm_ST/100.0,S6ST_)
    #print(P, S6ST_)

    if bulk_composition == "yes":
        wt_g = 0.
    elif bulk_composition == "wtg":
        wt_g = setup.loc[run,"wt_g"]/100.

    wt_S = species.loc["S","M"]*((wt_g*(((mg.xg_SO2(run,PT,melt_wf,setup,species,models)+2.0*mg.xg_S2(run,PT,melt_wf,setup,species,models))/mg.Xg_tot(run,PT,melt_wf,setup,species,models)) - (wm_ST/species.loc["S","M"]))) + (wm_ST/species.loc["S","M"]))
    
    Wt = 1.

    if eq_Fe == "no":
        wt_Fe = 0.0
    elif eq_Fe == "yes":
        total_dissolved_volatiles = (wm_ST*(1.-S6ST_) + (species.loc["SO3","M"]*((wm_ST*S6ST_)/species.loc["S","M"])))
        wt_Fe = (1.-wt_g)*(((1.0-total_dissolved_volatiles)*mg.wm_Fe_nv(run,melt_wf,setup,species))/100.0) # wt fraction of Fe

    wt_O = species.loc["O","M"]*((wt_g*(((2.0*mg.xg_O2(run,PT,melt_wf,setup,species,models) + 2.0*mg.xg_SO2(run,PT,melt_wf,setup,species,models))/mg.Xg_tot(run,PT,melt_wf,setup,species,models)) - (3.0*(wm_ST*S6ST_)/species.loc["S","M"]))) + (3.0*(wm_ST*S6ST_)/species.loc["S","M"]) + (wt_Fe/species.loc["Fe","M"])*((1.5*Fe3Fe2_+1.0)/(Fe3Fe2_+1.0)))
    return wt_O, wt_S, wt_Fe, wt_g, Wt

# calculate weight fraction of elements in the system when adding gas into a melt
def new_bulk_regas_open(run,PT,melt_wf,bulk_wf,gas_mf,dwtg,setup,species,models):
    wm_S, wm_Fe, wm_O = mg.melt_elements(run,PT,melt_wf,bulk_wf,gas_mf,setup,species,models)
    wg_O, wg_S, = mg.gas_elements(gas_mf,species)
    wt_O = (1.-dwtg)*wm_O + dwtg*wg_O
    wt_S = (1.-dwtg)*wm_S + dwtg*wg_S
    wt_Fe = (1.-dwtg)*wm_Fe
    Wt = bulk_wf['Wt']*(1. + dwtg)
    return wt_S, wt_Fe, wt_O, Wt

# mole fraction of elements in different species
def mf_S_species(melt_wf,gas_mf,species):
    # weight of S in each sulphur-bearing species
    W_S2m = melt_wf["ST"]*(1.-gas_mf["wt_g"])*(1.-melt_wf["S6ST"])
    W_SO4 = melt_wf["ST"]*(1.-gas_mf["wt_g"])*melt_wf["S6ST"]
    W_SO2 = ((gas_mf["SO2"]*species.loc["S","M"])/(gas_mf["Xg_t"]))*gas_mf["wt_g"]
    W_S2 = ((gas_mf["S2"]*species.loc["S2","M"])/(gas_mf["Xg_t"]))*gas_mf["wt_g"]
    W_total = W_S2m + W_SO4 + W_SO2 + W_S2
    # weight and mole fraction of S in each sulphur-bearing species compared to total S
    w_S2m = W_S2m/W_total
    w_SO4 = W_SO4/W_total
    w_SO2 = W_SO2/W_total
    w_S2 = W_S2/W_total
    mf_S = {"S2-":w_S2m, "SO42-":w_SO4, "SO2":w_SO2, "S2": w_S2}
    return mf_S


#########################
### sulphur satuation ###
#########################

# check solid/immiscible liquid sulphur saturation
def sulphur_saturation(run,PT,melt_wf,setup,species,models): # melt weight fraction of ST and S6/ST
    wmST = melt_wf['ST']
    S6T = mg.S6ST(run,PT,melt_wf,setup,species,models)
    wmS2 = wmST*100.0*10000.0*(1.0-S6T)
    wmS6 = wmST*100.0*10000.0*S6T
    SCSS_ = mg.SCSS(run,PT,melt_wf,setup,species,models)
    SCAS_ = mg.SCAS(run,PT,melt_wf,setup,species)
    StCSS = SCSS_/(1.-S6T)
    StCAS = SCAS_/S6T
    if wmS2 < SCSS_ and wmS6 < SCAS_:
        sulphide_sat = "no"
        sulphate_sat = "no"
        ST = wmST*1000000.
    elif wmS2 >= SCSS_ and wmS6 >= SCAS_:
        sulphide_sat = "yes"
        sulphate_sat = "yes"
        ST = min(StCSS,StCAS)
    elif wmS2 >= SCSS_ and wmS6 < SCAS_:
        sulphide_sat = "yes"
        sulphate_sat = "no"
        ST = StCSS
    elif wmS2 < SCSS_ and wmS6 >= SCAS_:
        sulphide_sat = "no"
        sulphate_sat = "yes"
        ST = StCAS
    else:
        sulphide_sat = "nan"
        sulphate_sat = "nan"
        ST = wmST*1000000.
    
    return SCSS_, sulphide_sat, SCAS_, sulphate_sat, ST


###############
### gassing ###
###############

# calculates equilibrium melt and gas composition for degassing calculation
def gassing(run,gassing_inputs,setup,species,models):
    
    print(setup.loc[run,"Sample"])
    
    # set T, volatile composition of the melt, and tolerances
    PT={"T":setup.loc[run,"T_C"]}
    melt_wf = {"ST":setup.loc[run,"STppm"]/1000000.}
    melt_wf["ST"] = melt_wf["ST"]
    nr_step = gassing_inputs["nr_step"]
    nr_tol = gassing_inputs["nr_tol"]
    dp_step = gassing_inputs["dp_step"]
    psat_tol = gassing_inputs["psat_tol"]
    dwtg = gassing_inputs["dwtg"]
    i_nr_step = gassing_inputs["i_nr_step"]
    i_nr_tol = gassing_inputs["i_nr_tol"]
    
    # Calculate saturation pressure
    P_sat_, wm_S2m_, wm_S6p_, S6p_ST, S2m_ST, = P_sat(run,PT,melt_wf,setup,species,models,psat_tol,nr_step,nr_tol)
    PT["P"] = P_sat_
    
    # update Fe3+/FeT and water speciation at saturation pressure and check for sulphur saturation
    melt_wf["Fe3FeT"] = mg.Fe3FeT_i(run,PT,setup,species,models)
    melt_wf["S6ST"] = mg.S6ST(run,PT,melt_wf,setup,species,models)
    SCSS_,sulphide_sat,SCAS_,sulphate_sat,ST_ = sulphur_saturation(run,PT,melt_wf,setup,species,models)
    
    # Set bulk composition
    wt_O, wt_S, wt_Fe, wt_g_, Wt_ = bulk_composition(run,PT,melt_wf,setup,species,models)
    bulk_wf = {"O":wt_O,"S":wt_S,"Fe":wt_Fe,"Wt":Wt_}
    
    # set system and initial guesses
    system = eq.set_system(melt_wf,models)
    guessx = eq.initial_guesses(run,PT,melt_wf,setup,species,models,system)
    
    # create results table
    results_header1 = pd.DataFrame([["System","run","Sample","ST (mwf)","Fe3FeT","S6ST","O (twf)","S (twf)","Fe (twf)","Saturation P (bars)",
"SiO2 (wt%)","TiO2 (wt%)","Al2O3 (wt%)","FeOT (wt%)","MnO (wt%)","MgO (wt%)","CaO (wt%)","Na2O (wt%)","K2O (wt%)","P2O5 (wt%)",
"oxygen fugacity","sulphide solubility","sulphate solubility","ideal gas","bulk composition","equilibrate Fe","starting pressure","gassing style","Date"]])
    results_header2 = pd.DataFrame([[system,run,setup.loc[run,"Sample"],melt_wf["ST"],melt_wf["Fe3FeT"],"SORT",bulk_wf["O"],bulk_wf["S"],bulk_wf["Fe"],P_sat_,
setup.loc[run,"SiO2"],setup.loc[run,"TiO2"],setup.loc[run,"Al2O3"],mg.Wm_FeOT(run,setup,species),setup.loc[run,"MnO"],setup.loc[run,"MgO"],setup.loc[run,"CaO"],setup.loc[run,"Na2O"],setup.loc[run,"K2O"],setup.loc[run,"P2O5"],
models.loc["fO2","option"],models.loc["sulphide","option"],models.loc["sulphate","option"],models.loc["ideal_gas","option"],models.loc["bulk_composition","option"],models.loc["eq_Fe","option"],models.loc["starting_P","option"],models.loc["gassing_style","option"],date.today()]])
    results_header = results_header1.append(results_header2, ignore_index=True)
    results_chemistry1 = pd.DataFrame([["P","T('C)","xg_O2","xg_S2","xg_SO2","Xg_t","wm_S","wm_SO3","wm_ST","Fe32","Fe3T","S62","S6T",
               "DFMQ","DNNO","SCSS","sulphide sat?","SCAS","sulphate sat?","wt_g","wt_g_O","wt_g_S","wt_O","wt_S",
               "fO2","fS2","fSO2","yO2","yS2","ySO2","C_S","C_SO4", "KD1","KSOg"]])
    results_chemistry = results_header.append(results_chemistry1, ignore_index=True)
    results1 = pd.DataFrame([[PT["P"],PT["T"],mg.xg_O2(run,PT,melt_wf,setup,species,models),mg.xg_S2(run,PT,melt_wf,setup,species,models),mg.xg_SO2(run,PT,melt_wf,setup,species,models),mg.Xg_tot(run,PT,melt_wf,setup,species,models),
(mg.wm_S(run,PT,melt_wf,setup,species,models)/100),(mg.wm_SO3(run,PT,melt_wf,setup,species,models)/100),melt_wf["ST"],mg.Fe3Fe2(melt_wf),melt_wf["Fe3FeT"],mg.S6S2(run,PT,melt_wf,setup,species,models),mg.S6ST(run,PT,melt_wf,setup,species,models),
mg.fO22Dbuffer(PT,mg.f_O2(run,PT,melt_wf,setup,species,models),"FMQ"),mg.fO22Dbuffer(PT,mg.f_O2(run,PT,melt_wf,setup,species,models),"NNO"),SCSS_,sulphide_sat,SCAS_,sulphate_sat,wt_g_,"","",
bulk_wf["O"],bulk_wf["S"],
mg.f_O2(run,PT,melt_wf,setup,species,models),mg.f_S2(run,PT,melt_wf,setup,species,models),mg.f_SO2(run,PT,melt_wf,setup,species,models),
mg.y_O2(PT,species,models),mg.y_S2(PT,species,models),mg.y_SO2(PT,species,models),
mg.C_S(run,PT,melt_wf,setup,species,models),mg.C_SO4(run,PT,melt_wf,setup,species,models),
mg.KD1(run,PT,setup,species,models),mg.KOSg(PT)]])
    results_chemistry = results_chemistry.append(results1, ignore_index=True)
    results_chemistry.to_csv('results_gassing_chemistry.csv', index=False, header=False)
    
    starting_P = models.loc["starting_P","option"]
    if starting_P == "set":
        initial = int(setup.loc[run,"P_bar"])
    else:
        initial = round(PT["P"])-1
    step = -1 # pressure step in bars
    final = 0
    
    # run over different pressures #
    for i in range(initial,final,step): # P is pressure in bars or T is temperature in 'C
        eq_Fe = models.loc["eq_Fe","option"]
        P = i/dp_step
        PT["P"] = P

        # work out equilibrium partitioning between melt and gas phase
        xg_O2_, xg_S2_, xg_SO2_, Xg_t, wm_S_, wm_SO3_, wm_ST_, Fe32, Fe3T, S62, S6T, wt_g_O, wt_g_S, wt_g, wt_O_, wt_S_, guessx = eq.mg_equilibrium(run,PT,melt_wf,bulk_wf,setup,species,models,nr_step,nr_tol,guessx)
        
        # set melt composition for forward calculation
        melt_wf["ST"] = wm_ST_
        melt_wf["Fe3FeT"] = Fe3T
        melt_wf["S6ST"] = S6T
        melt_wf["S2-"] = wm_S_
    
        # check for sulphur saturation and display warning in outputs
        SCSS_,sulphide_sat,SCAS_,sulphate_sat, ST_ = sulphur_saturation(run,PT,melt_wf,setup,species,models)
        if sulphide_sat == "yes":
            warning = "WARNING: sulphide-saturated"
        elif sulphate_sat == "yes":
            warning = "WARNING: sulphate-saturated"
        else:
            warning = ""
        
        # calculate fO2
        if eq_Fe == "yes":
            fO2_ = mg.f_O2(run,PT,melt_wf,setup,species,models)
        elif eq_Fe == "no":
            fO2_ = (xg_O2_*mg.y_O2(PT,models)*PT["P"])
            
        # volume, density, and mass
        gas_mf = {"O2":xg_O2_,"S2":xg_S2_,"SO2":xg_SO2_,"Xg_t":Xg_t,"wt_g":wt_g}
        
        # store results
        results2 = pd.DataFrame([[PT["P"],PT["T"],xg_O2_,xg_S2_,xg_SO2_,Xg_t,wm_S_,wm_SO3_,wm_ST_,Fe32,Fe3T,S62,S6T,
               mg.fO22Dbuffer(PT,fO2_,"FMQ"),mg.fO22Dbuffer(PT,fO2_,"NNO"),SCSS_,sulphide_sat,SCAS_,sulphate_sat,
               wt_g,wt_g_O,wt_g_S,wt_O_,wt_S_,
fO2_,mg.f_S2(run,PT,melt_wf,setup,species,models),mg.f_SO2(run,PT,melt_wf,setup,species,models),
mg.y_O2(PT,species,models),mg.y_S2(PT,species,models),mg.y_SO2(PT,species,models),
mg.C_S(run,PT,melt_wf,setup,species,models),mg.C_SO4(run,PT,melt_wf,setup,species,models),
mg.KD1(run,PT,setup,species,models),mg.KOSg(PT)]])
        results_chemistry = results_chemistry.append(results2, ignore_index=True)
        results_chemistry.to_csv('results_gassing_chemistry.csv', index=False, header=False)
                
        print(PT["T"],PT["P"],mg.fO22Dbuffer(PT,fO2_,"FMQ"),warning)

        # recalculate bulk composition if needed
        if models.loc["gassing_style","option"] == "open":
            wm_S_, wm_Fe_, wm_O_ = mg.melt_elements(run,PT,melt_wf,bulk_wf,gas_mf,setup,species,models)
            Wt_ = bulk_wf['Wt']
            bulk_wf = {"O":wm_O_,"S":wm_S_,"Fe":wm_Fe_,"Wt":(Wt_*(1. - wt_g))}
            
        
############################################
### melt-vapour equilibrium at given fO2 ###
############################################

def eq_given_fO2(inputs,setup,species,models): # only S atm
    
    option = inputs["option"]
    
    if option == "loop":
        run = inputs["run"]
        PT={'T':setup.loc[run,"T_C"]}
        melt_wf = {"ST":setup.loc[run,"STppm"]/1000000.}
        step = inputs["dfO2_step"]
        initial = inputs["fO2_i"]
        final = inputs["fO2_f"]
        start = 0
        end = inputs["no_steps"]
        difference = (final - initial) + 1
        step_size = difference/end
        # Set bulk composition
        wt_S = melt_wf["ST"]
        bulk_wf = {"S":wt_S}
        system = eq.set_system(bulk_wf,models)
        
    elif option == "spreadsheet":
        start = inputs["first row"]
        end = inputs["last row"]

    # create results table
    results_header1 = pd.DataFrame([["oxygen fugacity","sulphide solubility","sulphur speciation","ideal gas","bulk composition","equilibrate Fe","starting pressure","gassing style","Date"]])
    results_header2 = pd.DataFrame([[
models.loc["fO2","option"],models.loc["sulphide","option"],models.loc["sulphate","option"],models.loc["ideal_gas","option"],models.loc["gassing_style","option"],date.today()]])
    results_header = results_header1.append(results_header2, ignore_index=True)
    results_chemistry1 = pd.DataFrame([["P","T('C)","System","run","ST (mwf)","Fe3FeT","S6ST","O (twf)","S (twf)","Fe (twf)",
"SiO2 (wt%)","TiO2 (wt%)","Al2O3 (wt%)","FeOT (wt%)","MnO (wt%)","MgO (wt%)","CaO (wt%)","Na2O (wt%)","K2O (wt%)","P2O5 (wt%)","xg_O2","xg_S2","xg_SO2","Xg_t","wm_S","wm_SO3","wm_ST","Fe3T","S6T",
               "DFMQ","DNNO","SCSS","sulphide sat?","SCAS","sulphate sat?","wt_g","wt_O","wt_S",
               "fO2","fS2","fSO2","yO2","yS2","ySO2","C_S","C_SO4","KD1","KSOg"]])
    results_chemistry = results_header.append(results_chemistry1, ignore_index=True) 
    
    
    # run over different fO2 #
    for i in range(start,end,1): 
        if option == "loop":
            i_ = (i*step_size)+initial # fO2 in log units 
            fO2_ = 10.**i_
            PT["P"] = inputs["P"]
            print(setup.loc[run,"Sample"],fO2_)
        elif option == "spreadsheet":
            run = i
            PT={'T':setup.loc[run,"T_C"]}
            melt_wf = {"ST":setup.loc[run,"STppm"]/1000000.}
            PT["P"] = setup.loc[run,"P_bar"]
            melt_wf['Fe3FeT'] = mg.Fe3FeT_i(run,PT,setup,species,models)
            fO2_ = mg.f_O2(run,PT,melt_wf,setup,species,models)
            # Set bulk composition
            wt_S = melt_wf["ST"]
            bulk_wf = {"S":wt_S,"ST":wt_S}
            system = eq.set_system(bulk_wf,models)
            print(setup.loc[run,"Sample"],fO2_,PT["P"])
        
        # work out equilibrium partitioning between melt and gas phase
        melt_wf["Fe3FeT"] = mg.fO22Fe3FeT(fO2_,run,PT,setup,species,models)
        Fe3T = melt_wf["Fe3FeT"]
        xg_O2_, xg_S2_, xg_SO2_, wm_S_, wm_SO3_, wm_ST_, S6T, Xg_t, wt_S_, wt_g = eq.S_P_fO2(run,PT,fO2_,melt_wf,setup,species,models)

                
        # set melt composition for forward calculation
        melt_wf = {"ST":wm_ST_,"S6ST":S6T,"Fe3FeT":Fe3T,"S2-":wm_S_}
    
        # check for sulphur saturation and display warning in outputs
        SCSS_,sulphide_sat,SCAS_,sulphate_sat, ST_ = sulphur_saturation(run,PT,melt_wf,setup,species,models)
        if sulphide_sat == "yes":
            warning = "WARNING: sulphide-saturated"
        elif sulphate_sat == "yes":
            warning = "WARNING: sulphate-saturated"
        else:
            warning = ""

        # store results       
        results2 = pd.DataFrame([[PT["P"],PT["T"],system,run,setup.loc[run,"Sample"],melt_wf["ST"],melt_wf["Fe3FeT"],"SORT","SORT",bulk_wf["S"],"SORT",
setup.loc[run,"SiO2"],setup.loc[run,"TiO2"],setup.loc[run,"Al2O3"],mg.Wm_FeOT(run,setup,species),setup.loc[run,"MnO"],setup.loc[run,"MgO"],setup.loc[run,"CaO"],setup.loc[run,"Na2O"],setup.loc[run,"K2O"],setup.loc[run,"P2O5"],xg_O2_,xg_S2_,xg_SO2_,Xg_t,wm_S_,wm_SO3_,wm_ST_,Fe3T,S6T, mg.fO22Dbuffer(PT,fO2_,"FMQ"),mg.fO22Dbuffer(PT,fO2_,"NNO"),SCSS_,sulphide_sat,SCAS_,sulphate_sat,
               wt_g,"SORT",wt_S,
fO2_,mg.f_S2(run,PT,melt_wf,setup,species,models),mg.f_SO2(run,PT,melt_wf,setup,species,models),
mg.y_O2(PT,species,models),mg.y_S2(PT,species,models),mg.y_SO2(PT,species,models),
mg.C_S(run,PT,melt_wf,setup,species,models),mg.C_SO4(run,PT,melt_wf,setup,species,models),
mg.KD1(run,PT,setup,species,models),mg.KOSg(PT)]])
        results_chemistry = results_chemistry.append(results2, ignore_index=True)
        results_chemistry.to_csv('results_fO2_chemistry.csv', index=False, header=False)