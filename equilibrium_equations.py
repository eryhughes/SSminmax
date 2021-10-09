# equilibrium_equations.py

##########################
### packages to import ###
##########################

import pandas as pd
import numpy as np
import differential_equations as de
import melt_gas as mg


###################
### SOFe system ###
###################

def set_system(melt_wf,models):
    sys = "SOFe"
    return sys


######################################################
### specitation of the melt at given P, T, and fO2 ###
######################################################

def melt_speciation(run,PT,melt_wf,setup,species,models,nr_step,nr_tol):
    system = set_system(melt_wf,models)
    wt_S = melt_wf['ST']
    M_S = species.loc['S','M']
    S6p_ST = mg.S6ST(run,PT,melt_wf,setup,species,models)
    S2m_ST = 1. - S6p_ST  
    wm_S2m_ = S2m_ST*wt_S
    wm_S6p_ = S6p_ST*wt_S                
    return wm_S2m_, wm_S6p_, S6p_ST, S2m_ST


######################################################################################
### speciation of the silicate melt and vapour at given P, T, and bulk composition ###
######################################################################################

def initial_guesses(run,PT,melt_wf,setup,species,models,system):
    starting_P = models.loc["starting_P","option"]
    if starting_P == "set":
        guessx = setup.loc[run,"xg_O2"]
    else:
        guessx = mg.xg_O2(run,PT,melt_wf,setup,species,models)
    return guessx

def newton_raphson(x0,constants,e1,step,eqs,deriv):
    # create results table
    results = pd.DataFrame([["guessx","diff","step"]])  
    results.to_csv('results_newtraph.csv', index=False, header=False)
    
    def dx(x,eqs):
        f_,wtg1,wtg2 = eqs(x)
        result =(abs(0-f_))
        return result
    
    delta1 = dx(x0,eqs)
    
    results1 = pd.DataFrame([[x0,delta1,step]]) 
    results = results.append(results1, ignore_index=True)
    results.to_csv('results_newtraph.csv', index=False, header=False)     

    while delta1 > e1:
        while x0 < 0.:
            results1 = pd.DataFrame([[x0,delta1,step]]) 
            results = results.append(results1, ignore_index=True)
            results.to_csv('results_newtraph.csv', index=False, header=False)     
            step = step/10.
            x0 = x0 - step*(f_/df_)
        f_,wtg1,wtg2 = eqs(x0)
        df_ = deriv(x0,constants)
        x0 = x0 - step*(f_/df_)
        delta1 = dx(x0,eqs)
        results1 = pd.DataFrame([[x0,delta1,step]]) 
        results = results.append(results1, ignore_index=True)
        results.to_csv('results_newtraph.csv', index=False, header=False)     
    return x0      

def eq_SOFe(run,PT,bulk_wf,melt_wf,species,setup,models,nr_step,nr_tol,guessx):
    P = PT["P"]
    wt_O = bulk_wf['O']
    wt_S = bulk_wf['S']
    wt_Fe = bulk_wf['Fe']    
    
    # equilibrium constants
    K1_ = mg.KOSg(PT)
    K2_ = mg.C_S(run,PT,melt_wf,setup,species,models)/1000000.
    K3_ = mg.C_SO4(run,PT,melt_wf,setup,species,models)/1000000.
    KD1_ = mg.KD1(run,PT,setup,species,models)
    KD2 = 0.4
    y = 0.3
   
    # fugacity coefficients
    y_SO2_ = mg.y_SO2(PT,species,models)
    y_O2_ = mg.y_O2(PT,species,models)
    y_S2_ = mg.y_S2(PT,species,models)
    
    # molecular masses
    M_S = species.loc['S','M']
    M_O = species.loc['O','M']
    M_Fe = species.loc['Fe','M']
    M_O2 = species.loc['O2','M']
    M_S2 = species.loc['S2','M']
    M_SO2 = species.loc['SO2','M']
    M_SO3 = species.loc['SO3','M']        
    M_FeO = species.loc['FeO','M']
    M_FeO15 = species.loc['FeO1.5','M']
    M_m_ = mg.M_m_ox(run,setup,species)
    
    constants = [P, wt_O, wt_S, wt_Fe, K1_, K2_, K3_, KD1_, KD2, y, y_SO2_, y_O2_, y_S2_, M_S, M_O, M_Fe, M_O2, M_S2, M_SO2, M_SO3, M_FeO, M_FeO15, M_m_]
     
    def mg_SOFe(xg_O2_):
        a = (y_SO2_**2)/(K1_**2*y_O2_**2*xg_O2_**2*y_S2_*P)
        b = 1.
        c = xg_O2_ - 1.
        xg_SO2_ = (-b + (b**2.-(4.*a*c))**0.5)/(2.*a)
        xg_S2_ = (((y_SO2_*xg_SO2_)/(K1_*y_O2_*xg_O2_))**2.)/(y_S2_*P)
        wm_S_ = K2_*(((y_S2_*xg_S2_)/(y_O2_*xg_O2_))**0.5)
        wm_SO3_ = M_SO3*((((y_S2_*xg_S2_)**0.5*(y_O2_*xg_O2_)**1.5*P**2.0)*K3_)/M_S)
        wm_ST_ = wm_S_ + ((M_S*wm_SO3_)/M_SO3)        
        Xg_t = xg_SO2_*M_SO2 + xg_S2_*M_S2 + xg_O2_*M_O2
        Fe32 = ((KD1_*(y_O2_*xg_O2_*P)**0.25)+(2.0*y*KD2*(KD1_**(2.0*y))*((y_O2_*xg_O2_*P)**(0.5*y))))/(1.0 + (1.0 - 2.0*y)*KD2*(KD1_**(2.0*y))*((y_O2_*xg_O2_*P)**(0.5*y)))
        Fe3T = Fe32/(1+Fe32)
        S62 = (wm_SO3_/M_SO3)/(wm_S_/M_S)
        S6T = S62/(1+S62)
        return xg_SO2_, xg_S2_, Xg_t, Fe32, Fe3T, wm_S_, wm_SO3_, S62, S6T, wm_ST_
    
    def mb_SOFe(xg_O2_):
        xg_SO2_, xg_S2_, Xg_t, Fe32, Fe3T, wm_S_, wm_SO3_, S62, S6T, wm_ST_ = mg_SOFe(xg_O2_)
        diff,wt_g_O,wt_g_S = f_SOFe(xg_O2_)
        wt_g = (wt_g_O+wt_g_S)/2
        wt_S_ = M_S*((wt_g*(((xg_SO2_+2.0*xg_S2_)/Xg_t) - (wm_S_/M_S) - (wm_SO3_/M_SO3)))+((wm_S_/M_S) + (wm_SO3_/M_SO3)))
        wt_O_ = M_O*((wt_g*(((2.0*xg_SO2_ + 2.0*xg_O2_)/Xg_t)-(3.0*wm_SO3_/M_SO3)))+(3.0*wm_SO3_/M_SO3) + (wt_Fe/M_Fe)*((1.5*Fe32+1.0)/(Fe32+1.0)))
        return wt_g, wt_O_, wt_S_
    
    def f_SOFe(xg_O2_):
        xg_SO2_, xg_S2_, Xg_t, Fe32, Fe3T, wm_S_, wm_SO3_, S62, S6T, wm_ST_ = mg_SOFe(xg_O2_)
        wt_g_S = ((wt_S/M_S) - (wm_S_/M_S) - (wm_SO3_/M_SO3))/(((xg_SO2_+2.0*xg_S2_)/Xg_t) - (wm_S_/M_S) - (wm_SO3_/M_SO3))
        wt_g_O = ((wt_O/M_O) - (3.0*wm_SO3_/M_SO3) - (wt_Fe/M_Fe)*((1.5*Fe32+1.0)/(Fe32+1.0)))/(((2.0*xg_SO2_ + 2.0*xg_O2_)/Xg_t) - (3.0*wm_SO3_/M_SO3))
        diff = wt_g_S - wt_g_O
        return diff,wt_g_O,wt_g_S
    
    def df_SOFe(xg_O2_,constants):
        result = de.SOFe_O2(xg_O2_,constants)
        return result
    
    xg_O2_ = newton_raphson(guessx,constants,nr_tol,nr_step,f_SOFe,df_SOFe)
    result1 = mg_SOFe(xg_O2_)
    result2 = f_SOFe(xg_O2_)
    result3 = mb_SOFe(xg_O2_)
    return xg_O2_, result1, result2, result3

def mg_equilibrium(run,PT,melt_wf,bulk_wf,setup,species,models,nr_step,nr_tol,guessx):
    xg_O2_,A,B,C = eq_SOFe(run,PT,bulk_wf,melt_wf,species,setup,models,nr_step,nr_tol,guessx) # SOFe system
    xg_SO2_, xg_S2_, Xg_t, Fe32, Fe3T, wm_S_, wm_SO3_, S62, S6T, wm_ST_ = A
    diff,wt_g_O,wt_g_S = B
    wt_g, wt_O_, wt_S_ = C
    guessx = xg_O2_
    return xg_O2_, xg_S2_, xg_SO2_, Xg_t, wm_S_, wm_SO3_, wm_ST_, Fe32, Fe3T, S62, S6T, wt_g_O, wt_g_S, wt_g, wt_O_, wt_S_, guessx


################################################
### Pvsat and wmST for given T, fO2, and fS2 ###
################################################

def p_tot_fO2_fS2(run,PT,melt_wf,setup,species,models): 
    melt_wf["Fe3FeT"] = mg.Fe3FeT_i(run,PT,setup,species,models)
    fO2 = mg.f_O2(run,PT,melt_wf,setup,species,models)
    fS2 = setup.loc[run,"fS2"]
    K1 = mg.KOSg(PT)
    K2 = mg.C_S(run,PT,melt_wf,setup,species,models)/1000000.
    K3 = mg.C_SO4(run,PT,melt_wf,setup,species,models)/1000000.
    yO2 = mg.y_O2(PT,species,models)
    yS2 = mg.y_S2(PT,species,models)
    ySO2 = mg.y_SO2(PT,species,models)
    
    fSO2 = K1*fS2**0.5*fO2
    wm_S2 = ((fS2/fO2)**0.5)*K2 # wf S2- in the melt
    wm_S6 = ((fS2*fO2**3)**0.5)*K3 # wf S6+ in the melt
    wm_ST = wm_S2 + wm_S6 # wf ST in the melt
    
    pS2 = fS2/yS2
    pO2 = fO2/yO2
    pSO2 = fSO2/ySO2
    
    P_tot = pS2 + pO2 + pSO2
    
    xgS2 = pS2/P_tot
    xgO2 = pO2/P_tot
    xgSO2 = pSO2/P_tot
    
    return P_tot, wm_ST, fSO2, wm_S2, wm_S6, pS2, pO2, pSO2, xgS2, xgO2, xgSO2


############################################
### fS2 and wmST for given T, fO2, and P ###
############################################

def S_P_fO2(run,PT,fO2,melt_wf,setup,species,models):
    P = PT["P"]
    
    # equilibrium constants
    K6_ = mg.KOSg(PT)
    K8_ = mg.C_S(run,PT,melt_wf,setup,species,models)/1000000.0
    K9_ = (mg.C_SO4(run,PT,melt_wf,setup,species,models)/1000000.0)
    KD1_ = mg.KD1(run,PT,setup,species,models)
    KD2 = 0.4
    y = 0.3
   
    # fugacity coefficients
    y_O2_ = mg.y_O2(PT,species,models)
    y_S2_ = mg.y_S2(PT,species,models)
    y_SO2_ = mg.y_SO2(PT,species,models)
    
    # molecular masses
    M_O = species.loc['O','M']
    M_S = species.loc['S','M']
    M_Fe = species.loc['Fe','M']
    M_O2 = species.loc['O2','M']
    M_S2 = species.loc['S2','M']
    M_SO2 = species.loc['SO2','M']
    M_SO3 = species.loc['SO3','M']
    
    xg_O2_ = fO2/(P*y_O2_)
    a = 1.
    b = (K6_*(y_S2_*P)**0.5*xg_O2_*y_O2_)/y_SO2_
    c = xg_O2_ - 1.
    x = (-b + (b**2 - 4.*a*c)**0.5)/(2.*a)
    xg_S2_ = x**2
    xg_SO2_ = (K6_*(xg_S2_*P*y_S2_)**0.5*(xg_O2_*P*y_O2_))/(y_SO2_*P)
    wm_S_ = K8_*((y_S2_*xg_S2_)/(y_O2_*xg_O2_))**0.5
    wm_SO3_ = (K9_*(y_S2_*xg_S2_*P)**0.5*(y_O2_*xg_O2_*P)**1.5)*(M_SO3/M_S)
    wm_ST_ = wm_S_ + ((M_S*wm_SO3_)/M_SO3)
    S62 = (wm_SO3_/M_SO3)/(wm_S_/M_S)
    S6T = S62/(1+S62)
    Xg_t = xg_SO2_*M_SO2 + xg_S2_*M_S2 + xg_O2_*M_O2
    
    wt_S_, wt_g  = "na", "na"

    return xg_O2_, xg_S2_, xg_SO2_, wm_S_, wm_SO3_, wm_ST_, S6T, Xg_t, wt_S_, wt_g