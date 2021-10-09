# melt-gas.py

##########################
### packages to import ###
##########################

import pandas as pd
import numpy as np
import gmpy2 as gp
import math


#########################
### sulphide capacity ###
#########################

def C_S(run,PT,melt_wf,setup,species,models): # 0.5S2(v) + O2-(m) = S2-(m) + 0.5O2(v)
    model = models.loc["sulphide","option"]
    T = PT['T'] + 273.15 # T in K
    
    if model == "ONeill20": # O'Neill, H.S. (2021). The Thermodynamic Controls on Sulfide Saturation in Silicate Melts with Application to Ocean Floor Basalts. In Magma Redox Geochemistry (eds R. Moretti and D.R. Neuville). https://doi.org/10.1002/9781119473206.ch10
               
        # Mole fractions in the melt on cationic lattice (all Fe as FeO) no volatiles
        tot = ((species.loc["SiO2","no_cat"]*setup.loc[run,"SiO2"])/species.loc["SiO2","M"]) + ((species.loc["TiO2","no_cat"]*setup.loc[run,"TiO2"])/species.loc["TiO2","M"]) + ((species.loc["Al2O3","no_cat"]*setup.loc[run,"Al2O3"])/species.loc["Al2O3","M"]) + ((species.loc["FeO","no_cat"]*Wm_FeOT(run,setup,species))/species.loc["FeO","M"]) + ((species.loc["MgO","no_cat"]*setup.loc[run,"MgO"])/species.loc["MgO","M"]) + ((species.loc["MnO","no_cat"]*setup.loc[run,"MnO"])/species.loc["MnO","M"]) + ((species.loc["CaO","no_cat"]*setup.loc[run,"CaO"])/species.loc["CaO","M"]) + ((species.loc["Na2O","no_cat"]*setup.loc[run,"Na2O"])/species.loc["Na2O","M"]) + ((species.loc["K2O","no_cat"]*setup.loc[run,"K2O"])/species.loc["K2O","M"]) + ((species.loc["P2O5","no_cat"]*setup.loc[run,"P2O5"])/species.loc["P2O5","M"]) 
        Si = ((species.loc["SiO2","no_cat"]*setup.loc[run,"SiO2"])/species.loc["SiO2","M"])/tot
        Ti = ((species.loc["TiO2","no_cat"]*setup.loc[run,"TiO2"])/species.loc["TiO2","M"])/tot
        Al = ((species.loc["Al2O3","no_cat"]*setup.loc[run,"Al2O3"])/species.loc["Al2O3","M"])/tot
        Fe = ((species.loc["FeO","no_cat"]*Wm_FeOT(run,setup,species))/species.loc["FeO","M"])/tot
        Mn = ((species.loc["MnO","no_cat"]*setup.loc[run,"MnO"])/species.loc["MnO","M"])/tot
        Mg = ((species.loc["MgO","no_cat"]*setup.loc[run,"MgO"])/species.loc["MgO","M"])/tot
        Ca = ((species.loc["CaO","no_cat"]*setup.loc[run,"CaO"])/species.loc["CaO","M"])/tot
        Na = ((species.loc["Na2O","no_cat"]*setup.loc[run,"Na2O"])/species.loc["Na2O","M"])/tot
        K = ((species.loc["K2O","no_cat"]*setup.loc[run,"K2O"])/species.loc["K2O","M"])/tot
      
        lnC = (8.77 - (23590.0/T) + (1673.0/T)*(6.7*(Na+K) + 4.9*Mg + 8.1*Ca + 8.9*(Fe+Mn) + 5.0*Ti + 1.8*Al - 22.2*Ti*(Fe+Mn) + 7.2*Fe*Si) - 2.06*math.erf(-7.2*(Fe+Mn)))

    C = math.exp(lnC)    
    return C


#########################
### sulphate capacity ###
#########################

def C_SO4(run,PT,melt_wf,setup,species,models): # 0.5S2(v) + 1.5O2(v) + O2-(m) = SO42-(m)
    model = models.loc["sulphate","option"]
    T = PT['T'] + 273.15 # T in Kelvin
    slope = 115619.707 # slope for T-dependence for melt inclusion fits
    if model == "Nash19":  # Nash, W.M., Smythe, D.J., Wood, B.J. (2019). Compositional and temperature effects on sulfur speciation and solubility in silicate melts. Earth and Planetary Science Letters 507:187-198. https://doi.org/10.1016/j.epsl.2018.12.006
        S = 1. # S6+/S2- ratio of S6+/S2- of 0.5
        Csulphide = C_S(run,PT,setup,species,models)
        A = PT_KCterm(run,PT,setup,species,models) # P, T, compositional term from Kress & Carmicheal (1991)
        B = (8743600/T**2) - (27703/T) + 20.273 # temperature dependence from Nash et al. (2019)
        a = 0.196 # alnfO2 from Kress & Carmicheal (1991)
        F = 10**(((math.log10(S))-B)/8.)
        fO2 = math.exp(((math.log(0.5*F))-A)/a)
        Csulphate = (S*Csulphide)/(fO2**2)
    elif model == "S6ST": # if S6+/ST is known, and fO2 (e.g., from Fe3+/FeT), can calculate CS6+ using known CS2-
        Csulphide = C_S(run,PT,setup,species,models)
        fO2 = f_O2(run,PT,melt_wf,setup,species,models)
        S6ST_ = melt_wf["S6ST"]
        S = overtotal2ratio(S6ST_)
        Csulphate = (S*Csulphide)/(fO2**2)
    elif model == "Hawaii": # this study - see details in Supplementary Material
        Csulphate = math.exp(slope*(1./T) -48.)
    return Csulphate


################
### fugacity ###
################

def f_S2(run,PT,melt_wf,setup,species,models): # wtppm S2- NOT mole fraction due to parameterisation by O'Neill (2020)
    K = C_S(run,PT,melt_wf,setup,species,models)/1000000.
    fS2 = ((melt_wf["S2-"]/K)**2.)*f_O2(run,PT,melt_wf,setup,species,models)
    return fS2

def f_SO2(run,PT,melt_wf,setup,species,models):
    K = KOSg(PT)
    return K*f_O2(run,PT,melt_wf,setup,species,models)*f_S2(run,PT,melt_wf,setup,species,models)**0.5


#######################
### oxygen fugacity ###
#######################

# buffers
def NNO(PT):
    P = PT['P']
    T_K = PT["T"]+273.15
    return (-24930/T_K + 9.36 + 0.046*(P-1.0)/T_K) # Frost, B.R. 1991. Introduction to oxygen fugacity and its petrologic importance. Reviews in Mineralogy and Geochemistry, 25, 1–9.
def FMQ(PT):
    P = PT['P']
    T_K = PT["T"]+273.15
    return (-25096.3/T_K + 8.735 + 0.11*(P-1.0)/T_K) # Frost, B.R. 1991. Introduction to oxygen fugacity and its petrologic importance. Reviews in Mineralogy and Geochemistry, 25, 1–9.
def fO22Dbuffer(PT,fO2,buffer):
    if buffer == "NNO":
        return math.log10(fO2) - NNO(PT)
    elif buffer == "FMQ":
        return math.log10(fO2) - FMQ(PT)
def Dbuffer2fO2(PT,D,buffer):
    if buffer == "NNO":
        return 10.0**(D + NNO(PT))
    elif buffer == "FMQ":
        return 10.0**(D + FMQ(PT))
    
# Compositional parameter for Kress & Carmichael (1991)
# Kress, V.C. and Carmichael, I.S.E. 1991. The compressibility of silicate liquids containing Fe2O3 and the effect of composition, temperature, oxygen fugacity and pressure on their redox states. Contributions to Mineralogy and Petrology, 108, 82–92, https://doi.org/10.1007/BF00307328.

def KC_mf(run,setup,species): # requires mole frations in the melt based on oxide components (all Fe as FeO) with no volatiles    
    tot = (setup.loc[run,"SiO2"]/species.loc["SiO2","M"]) + (setup.loc[run,"TiO2"]/species.loc["TiO2","M"]) + (setup.loc[run,"Al2O3"]/species.loc["Al2O3","M"]) + (Wm_FeOT(run,setup,species)/species.loc["FeO","M"]) + (setup.loc[run,"MnO"]/species.loc["MnO","M"]) + (setup.loc[run,"MgO"]/species.loc["MgO","M"]) + (setup.loc[run,"CaO"]/species.loc["CaO","M"]) + (setup.loc[run,"Na2O"]/species.loc["Na2O","M"]) + (setup.loc[run,"K2O"]/species.loc["K2O","M"])  + (setup.loc[run,"P2O5"]/species.loc["P2O5","M"])
    Al = (setup.loc[run,"Al2O3"]/species.loc["Al2O3","M"])/tot
    Fe = (Wm_FeOT(run,setup,species)/species.loc["FeO","M"])/tot
    Ca = (setup.loc[run,"CaO"]/species.loc["CaO","M"])/tot
    Na = (setup.loc[run,"Na2O"]/species.loc["Na2O","M"])/tot
    K = (setup.loc[run,"K2O"]/species.loc["K2O","M"])/tot
    return Al, Fe, Ca, Na, K

def d4X_KC(run,setup,species,models):
    Al, Fe, Ca, Na, K = KC_mf(run,setup,species)
    DAl = -2.243
    DFe = -1.828
    DCa = 3.201
    DNa = 5.854
    DK = 6.215
    return DAl*Al + DFe*Fe + DCa*Ca + DNa*Na + DK*K

def d4X_KCA(run,setup,species,models):
    Al, Fe, Ca, Na, K = KC_mf(run,setup,species)
    DWAl = 39.86e3             #J
    DWCa = -62.52e3            #J
    DWNa = -102.0e3            #J
    DWK = -119.0e3             #J
    return DWAl*Al+DWCa*Ca+DWNa*Na+DWK*K

def PT_KCterm(run,PT,setup,species,models):
    P = PT['P']
    T_K = PT['T']+273.15
    b = 1.1492e4 # K
    c = -6.675
    D4X = d4X_KC(run,setup,species,models)
    e = -3.36
    f = -7.01e-7 # K/Pa
    g = -1.54e-10 # /Pa
    h = 3.85e-17 # K/Pa2
    T0 = 1673.0 # K
    P_Pa = P*1.0e5 # converts bars to pascals
    value = (b/T_K) + c + D4X + e*(1.0 - (T0/T_K) - math.log(T_K/T0)) + f*(P_Pa/T_K) + g*(((T_K-T0)*P_Pa)/T_K) + h*((P_Pa**2.0)/T_K)
    return value

def KC91(run,PT,melt_wf,setup,species,models):    
    F = 0.5*Fe3Fe2(melt_wf) # XFe2O3/XFeO
    a = 0.196
    PTterm = PT_KCterm(run,PT,setup,species,models)
    alnfO2 = math.log(F) - PTterm
    return math.exp(alnfO2/a)

def KD1(run,PT,setup,species,models): #K&C91 appendix A 
    T_K = PT['T']+273.15
    P = PT['P']
    DH = -106.2e3               #J
    DS = -55.10                 #J/K
    DCp = 31.86                 #J/K
    DV = 7.42e-6                #m3
    DVdot = 1.63e-9             #m3/K
    DVdash = -8.16e-16          #m3/Pa
    D4X = d4X_KCA(run,setup,species,models)
    T0 = 1673.0                 # K
    P0 = 1.0e5                  # Pa 
    R = 8.3144598               # J/K/mol
    P_Pa = P*1.0e5
    return math.exp((-DH/(R*T_K)) + (DS/R) - (DCp/R)*(1.0 - (T0/T_K) - gp.log(T_K/T0)) - (1.0/(R*T_K))*D4X - ((DV*(P_Pa-P0))/(R*T_K)) - ((DVdot*(T_K-T0)*(P_Pa-P0))/(R*T_K)) - (DVdash/(2.0*R*T_K))*pow((P_Pa-P0),2.0))

def f_O2(run,PT,melt_wf,setup,species,models): 
    model = models.loc["fO2","option"]
    
    if model == "yes":
        return 10.0**(setup.loc[run,"logfO2"]) 

    elif model == "Kress91": # Kress & Carmichael (1991) equation/table 7
        fO2 = KC91(run,PT,melt_wf,setup,species,models)
        return fO2
    
    elif model == "Kress91A": # Kress & Carmichael (1991) equations A5-6 and table A1
        F = Fe3Fe2(melt_wf) # XFeO1.5/XFeO
        D4X = d4X_KCA(run,setup,species,models)
        KD2 = 0.4
        y = 0.3
        kd1 = KD1(run,PT,setup,species,models)
            
        def f(y,F,KD2,kd1,x): # KC91A rearranged to equal 0
            f = ((2.0*y - F + 2.0*y*F)*KD2*kd1**(2.0*y)*x**(0.5*y) + kd1*x**0.25 - F)
            return f

        def df(y,F,KD2,kd1,x): # derivative of above
            df = (0.5*y)*(2.0*y - F +2.0*y*F)*KD2*kd1**(2.0*y)*x**((0.5*y)-1.0) + 0.25*kd1*x**-0.75
            return df

        def dx(x):
            diff = abs(0-f(y,F,KD2,kd1,x))
            return diff
 
        def nr(x0, e1):
            delta1 = dx(x0)
            while delta1 > e1:
                x0 = x0 - f(y,F,KD2,kd1,x0)/df(y,F,KD2,kd1,x0)
                delta1 = dx(x0)
            return x0
            
        x0 = KC91(run,PT,melt_wf,setup,species,models)
    
        fO2 = nr(x0, 1e-15)
        return fO2
        
    elif model == "ONeill18": # O'Neill, H.S.C., Berry, A.J., Mallmann, G. (2018). The oxidation state of iron in Mid-Ocean Ridge Basaltic (MORB) glasses: Implications for their petrogenesis and oxygen fugacities Earth and Planetary Science Letters 504:152-162. https://doi.org/10.1016/j.epsl.2018.10.002
        F = Fe3Fe2(melt_wf) # Fe3+/Fe2+
        # mole fractions on a single cation basis
        tot = ((species.loc["SiO2","no_cat"]*setup.loc[run,"SiO2"])/species.loc["SiO2","M"]) + ((species.loc["TiO2","no_cat"]*setup.loc[run,"TiO2"])/species.loc["TiO2","M"]) + ((species.loc["Al2O3","no_cat"]*setup.loc[run,"Al2O3"])/species.loc["Al2O3","M"]) + ((species.loc["FeO","no_cat"]*Wm_FeOT(run,setup,species))/species.loc["FeO","M"]) + ((species.loc["MgO","no_cat"]*setup.loc[run,"MgO"])/species.loc["MgO","M"]) + ((species.loc["MnO","no_cat"]*setup.loc[run,"MnO"])/species.loc["MnO","M"]) + ((species.loc["CaO","no_cat"]*setup.loc[run,"CaO"])/species.loc["CaO","M"]) + ((species.loc["Na2O","no_cat"]*setup.loc[run,"Na2O"])/species.loc["Na2O","M"]) + ((species.loc["K2O","no_cat"]*setup.loc[run,"K2O"])/species.loc["K2O","M"]) + ((species.loc["P2O5","no_cat"]*setup.loc[run,"P2O5"])/species.loc["P2O5","M"])
        Ca = ((species.loc["CaO","no_cat"]*setup.loc[run,"CaO"])/species.loc["CaO","M"])/tot
        Na = ((species.loc["Na2O","no_cat"]*setup.loc[run,"Na2O"])/species.loc["Na2O","M"])/tot
        K = ((species.loc["K2O","no_cat"]*setup.loc[run,"K2O"])/species.loc["K2O","M"])/tot
        P = ((species.loc["P2O5","no_cat"]*setup.loc[run,"P2O5"])/species.loc["P2O5","M"])/tot
        DQFM = (math.log(F) + 1.36 - 2.4*Ca - 2.0*Na - 3.7*K + 2.4*P)/0.25
        logfO2 = DQFM + (8.58 - (25050/T_K)) # O'Neill, H.S.C. (1987). Quartz–fayalite–iron and quartz–fayalite–magnetite equilibria and the free energy of formation of fayalite (Fe2SiO4) and magnetite (Fe3O4). American Mineralogist 72(1-2):67-75
        return 10.0**logfO2
    
    elif model == "S6ST":
        S6T = melt_wf['S6ST']
        S62 = overtotal2ratio(S6T)
        fO2 = ((S62*C_S(run,PT,setup,species,models))/C_SO4(run,PT,melt_wf,setup,species,models))**0.5
        return fO2

    
#############################        
### fugacity coefficients ###
#############################

# all fugacity coefficients are assumed to equal 1 below 1 bar.
# Shi, P. and Saxena, S.K. 1992. Thermodynamic modeing of the C-H-O-S fluid system. American Mineralogist, 77, 1038–1049.
def lny_SS(PT,Pcr,Tcr):
    P = PT['P']
    T_K = PT['T']+273.15
    Tr = T_K/Tcr
    A, B, C, D, P0, integral0 = Q_SS(PT,Tr,Pcr)
    Pr = P/Pcr
    P0r = P0/Pcr
    integral = A*gp.log(Pr/P0r) + B*(Pr - P0r) + (C/2.0)*(pow(Pr,2.0) - pow(P0r,2.0)) + (D/3.0)*(pow(Pr,3.0) - pow(P0r,3.0))
    integral_total = integral + integral0
    return integral_total

def Q_SS(PT,Tr,Pcr):
    P = PT['P']
    def Q1000(Pcr):
        Pr_ = 1000.0/Pcr
        P0r_ = 1.0/Pcr
        A0 = 1.0
        B0 = 0.9827e-1*pow(Tr,-1.0) + -0.2709*pow(Tr,-3.0)
        C0 = -0.1030e-2*pow(Tr,-1.5) + 0.1427e-1*pow(Tr,-4.0)
        D0 = 0.0
        return A0*gp.log(Pr_/P0r_) + B0*(Pr_ - P0r_) + (C0/2.0)*(pow(Pr_,2.0) - pow(P0r_,2.0)) + (D0/3.0)*(pow(Pr_,3.0) - pow(P0r_,3.0))
    def Q5000(Pcr):
        Pr_ = 5000.0/Pcr
        P0r_ = 1000.0/Pcr
        A0 = 1.0 + -5.917e-1*pow(Tr,-2.0)
        B0 = 9.122e-2*pow(Tr,-1.0)
        C0 = -1.416e-4*pow(Tr,-2.0) + -2.835e-6*gp.log(Tr)
        D0 = 0.0
        return A0*gp.log(Pr_/P0r_) + B0*(Pr_ - P0r_) + (C0/2.0)*(pow(Pr_,2.0) - pow(P0r_,2.0)) + (D0/3.0)*(pow(Pr_,3.0) - pow(P0r_,3.0))
    if P > 5000.0:
        A = 2.0614 + -2.235*pow(Tr,-2.0) + -3.941e-1*gp.log(Tr)
        B = 5.513e-2*pow(Tr,-1.0) + 3.934e-2*pow(Tr,-2.0)
        C = -1.894e-6*pow(Tr,-1.0) + -1.109e-5*pow(Tr,-2.0) + -2.189e-5*pow(Tr,-3.0)
        D = 5.053e-11*pow(Tr,-1.0) + -6.303e-21*pow(Tr,3.0)
        P0 = 5000.0
        integral0 = Q1000(Pcr) + Q5000(Pcr)
        return A, B, C, D, P0, integral0
    elif P == 5000.0:
        A = 0
        B = 0
        C = 0
        D = 0
        P0 = 5000.0
        integral0 = Q1000(Pcr) + Q5000(Pcr)
        return A, B, C, D, P0, integral0
    elif P > 1000.0 and P < 5000.0:
        A = 1.0 + -5.917e-1*pow(Tr,-2.0)
        B = 9.122e-2*pow(Tr,-1.0)
        C = -1.416e-4*pow(Tr,-2.0) + -2.835e-6*gp.log(Tr)
        D = 0.0
        P0 = 1000.0
        integral0 = Q1000(Pcr)
        return A, B, C, D, P0, integral0
    elif P == 1000.0:
        A = 0
        B = 0
        C = 0
        D = 0.0
        P0 = 1000.0
        integral0 = Q1000(Pcr)
        return A, B, C, D, P0, integral0
    else:
        A = 1.0
        B = 0.9827e-1*pow(Tr,-1.0) + -0.2709*pow(Tr,-3.0)
        C = -0.1030e-2*pow(Tr,-1.5) + 0.1427e-1*pow(Tr,-4.0)
        D = 0.0
        P0 = 1.0
        integral0 = 0.0
        return A, B, C, D, P0, integral0
    

def y_SS(gas_species,PT,species,models):
    P = PT['P']
    T_K = PT['T']+273.15
    ideal_gas = models.loc["ideal_gas","option"]
    if ideal_gas == "yes":
        return 1.0
    elif P < 1.: # ideal gas below 1 bar
        return 1.
    else:    
        Tcr = species.loc[gas_species,"Tcr"]
        Pcr = species.loc[gas_species,"Pcr"]
        return gp.exp(lny_SS(PT,Pcr,Tcr))/P    

def y_O2(PT,species,models):
    gas_species = "O2"
    y = y_SS(gas_species,PT,species,models)
    return y
    
def y_S2(PT,species,models):
    gas_species = "S2"
    y = y_SS(gas_species,PT,species,models)
    return y

def y_SO2(PT,species,models): # option to modify below 500 bars
    P = PT['P']
    T_K = PT['T']+273.15
    ideal_gas = models.loc["ideal_gas","option"]
    gas_species = "SO2"
    if ideal_gas == "yes":
        return 1.
    elif P < 1.: # ideal gas below 1 bar
        return 1.
    else: # 1-10000 bar
        Tcr = species.loc[gas_species,"Tcr"] # critical temperature in K
        Pcr = species.loc[gas_species,"Pcr"] # critical temperature in bar
        P0 = 1.0
        P0r = P0/Pcr
        Tr = T_K/Tcr
        Q1_A, Q2_A, Q3_A, Q4_A, Q5_A, Q6_A, Q7_A, Q8_A  = 0.92854, 0.43269e-1, -0.24671, 0., 0.24999, 0., -0.53182, -0.16461e-1
        Q1_B, Q2_B, Q3_B, Q4_B, Q5_B, Q6_B, Q7_B, Q8_B  = 0.84866e-3, -0.18379e-2, 0.66787e-1, 0., -0.29427e-1, 0., 0.29003e-1, 0.54808e-2
        Q1_C, Q2_C, Q3_C, Q4_C, Q5_C, Q6_C, Q7_C, Q8_C  = -0.35456e-3, 0.23316e-4, 0.94159e-3, 0., -0.81653e-3, 0., 0.23154e-3, 0.55542e-4
        A = Q1_A + Q2_A*Tr + Q3_A*Tr**(-1.) + Q4_A*Tr**2. + Q5_A*Tr**(-2.) + Q6_A*Tr**3. + Q7_A*Tr**(-3.0) + Q8_A*gp.log(Tr)
        B = Q1_B + Q2_B*Tr + Q3_B*Tr**(-1.) + Q4_B*Tr**2. + Q5_B*Tr**(-2.) + Q6_B*Tr**3. + Q7_B*Tr**(-3.0) + Q8_B*gp.log(Tr)
        C = Q1_C + Q2_C*Tr + Q3_C*Tr**(-1.) + Q4_C*Tr**2. + Q5_C*Tr**(-2.) + Q6_C*Tr**3. + Q7_C*Tr**(-3.0) + Q8_C*gp.log(Tr)
        D = 0.0
        if P >= 500.: # above 500 bar using Shi and Saxena (1992) as is
            Pr = P/Pcr
            integral = A*gp.log(Pr/P0r) + B*(Pr - P0r) + (C/2.0)*(pow(Pr,2.0) - pow(P0r,2.0)) + (D/3.0)*(pow(Pr,3.0) - pow(P0r,3.0))
            return (gp.exp(integral))/P
        elif models.loc["y_SO2","option"] == "SS92": # as is Shi and Saxena (1992)
            Pr = P/Pcr
            integral = A*gp.log(Pr/P0r) + B*(Pr - P0r) + (C/2.0)*(pow(Pr,2.0) - pow(P0r,2.0)) + (D/3.0)*(pow(Pr,3.0) - pow(P0r,3.0))
            return (gp.exp(integral))/P
        elif models.loc["y_SO2","option"] == "SS92_modified": # below 500 bar linear fit between the value at 500 bar and y = 1 at 1 bar to avoid weird behaviour...
            Pr = 500./Pcr # calculate y at 500 bar
            integral = A*gp.log(Pr/P0r) + B*(Pr - P0r) + (C/2.0)*(pow(Pr,2.0) - pow(P0r,2.0)) + (D/3.0)*(pow(Pr,3.0) - pow(P0r,3.0))
            y_500 = (gp.exp(integral))/500.
            y = ((y_500 - 1.)*(P/500.)) + 1. # linear extrapolation to P of interest
            return y       

        
########################        
### partial pressure ###
########################

def p_O2(run,PT,melt_wf,setup,species,models):
    return f_O2(run,PT,melt_wf,setup,species,models)/y_O2(PT,species,models)
def p_SO2(run,PT,melt_wf,setup,species,models):
    return f_SO2(run,PT,melt_wf,setup,species,models)/y_SO2(PT,species,models)
def p_S2(run,PT,melt_wf,setup,species,models):
    return f_S2(run,PT,melt_wf,setup,species,models)/y_S2(PT,species,models)
def p_tot(run,PT,melt_wf,setup,species,models):
    return p_O2(run,PT,melt_wf,setup,species,models) + p_SO2(run,PT,melt_wf,setup,species,models) + p_S2(run,PT,melt_wf,setup,species,models)


######################       
### molar fraction ###
######################

def xg_O2(run,PT,melt_wf,setup,species,models):
    P = PT['P']
    return p_O2(run,PT,melt_wf,setup,species,models)/P
def xg_SO2(run,PT,melt_wf,setup,species,models):
    P = PT['P']
    return p_SO2(run,PT,melt_wf,setup,species,models)/P
def xg_S2(run,PT,melt_wf,setup,species,models):
    P = PT['P']
    return p_S2(run,PT,melt_wf,setup,species,models)/P
def Xg_tot(run,PT,melt_wf,setup,species,models):
    P = PT['P']
    Xg_t = xg_O2(run,PT,melt_wf,setup,species,models)*species.loc["O2","M"] + xg_SO2(run,PT,melt_wf,setup,species,models)*species.loc["SO2","M"] + xg_S2(run,PT,melt_wf,setup,species,models)*species.loc["S2","M"]   
    return Xg_t


#############################
### equilibrium constants ###
#############################

# 0.5S2 + O2 = SO2
# K = fSO2/((fS2^0.5)*fO2)
# Ohmoto, H. and Kerrick, D.M. 1977. Devolatilization equilibria in graphitic systems. American Journal of Science, 277, 1013–1044, https://doi.org/10.2475/AJS.277.8.1013.
def KOSg(PT):
    T_K = PT['T']+273.15
    return 10.**((18929.0/T_K)-3.783)

#####################
### mole fraction ###
#####################

# totals
def wm_vol(melt_wf): # wt% total volatiles in the melt
    wm_H2OT = 100.*melt_wf["H2OT"]
    wm_CO2 = 100.*melt_wf["CO2"]
    return wm_H2OT + wm_CO2 #+ wm_S(wm_ST) + wm_SO3(wm_ST,species)
def wm_nvol(melt_wf): # wt% total of non-volatiles in the melt
    return 100.0 - wm_vol(melt_wf)

# molecular mass on a singular oxygen basis
def M_m_SO(run,setup,species): # no volatiles
    Wm_tot = setup.loc[run,"SiO2"] + setup.loc[run,"TiO2"] + setup.loc[run,"Al2O3"] + setup.loc[run,"MnO"] + setup.loc[run,"MgO"] + setup.loc[run,"MnO"] + setup.loc[run,"CaO"] + setup.loc[run,"Na2O"] + setup.loc[run,"K2O"] + setup.loc[run,"P2O5"] + Wm_FeOT(run,setup,species)
    Xm_tot = (setup.loc[run,"SiO2"]/(species.loc["SiO2","M"]/species.loc["SiO2","no_O"])) + (setup.loc[run,"TiO2"]/(species.loc["TiO2","M"]/species.loc["TiO2","no_O"])) + (setup.loc[run,"Al2O3"]/(species.loc["Al2O3","M"]/species.loc["Al2O3","no_O"])) + (setup.loc[run,"MnO"]/(species.loc["MnO","M"]/species.loc["MnO","no_O"])) + (setup.loc[run,"MgO"]/(species.loc["MgO","M"]/species.loc["MgO","no_O"])) + (setup.loc[run,"CaO"]/(species.loc["CaO","M"]/species.loc["CaO","no_O"])) + (setup.loc[run,"Na2O"]/(species.loc["Na2O","M"]/species.loc["Na2O","no_O"])) + (setup.loc[run,"K2O"]/(species.loc["K2O","M"]/species.loc["K2O","no_O"])) + (setup.loc[run,"P2O5"]/(species.loc["P2O5","M"]/species.loc["P2O5","no_O"])) + (Wm_FeOT(run,setup,species)/(species.loc["FeO","M"]/species.loc["FeO","no_O"]))   
    result = Wm_tot/Xm_tot
    return result

# molecular mass on a oxide basis
def M_m_ox(run,setup,species): # no volatiles
    Wm_tot = setup.loc[run,"SiO2"] + setup.loc[run,"TiO2"] + setup.loc[run,"Al2O3"] + setup.loc[run,"MnO"] + setup.loc[run,"MgO"] + setup.loc[run,"MnO"] + setup.loc[run,"CaO"] + setup.loc[run,"Na2O"] + setup.loc[run,"K2O"] + setup.loc[run,"P2O5"] + Wm_FeOT(run,setup,species)
    Xm_tot = (setup.loc[run,"SiO2"]/species.loc["SiO2","M"]) + (setup.loc[run,"TiO2"]/species.loc["TiO2","M"]) + (setup.loc[run,"Al2O3"]/species.loc["Al2O3","M"]) + (setup.loc[run,"MnO"]/species.loc["MnO","M"]) + (setup.loc[run,"MgO"]/species.loc["MgO","M"]) + (setup.loc[run,"CaO"]/species.loc["CaO","M"]) + (setup.loc[run,"Na2O"]/species.loc["Na2O","M"]) + (setup.loc[run,"K2O"]/species.loc["K2O","M"]) + (setup.loc[run,"P2O5"]/species.loc["P2O5","M"]) + (Wm_FeOT(run,setup,species)/species.loc["FeO","M"])   
    result = Wm_tot/Xm_tot
    return result
    
# Number of moles in the melt
def Xm_H2OT(melt_wf,species):
    wm_H2OT = 100.*melt_wf['H2OT']
    return wm_H2OT/species.loc["H2O","M"]
def Xm_CO2(melt_wf,species):
    wm_CO2 = 100.*melt_wf['CO2']
    return wm_CO2/species.loc["CO2","M"]
def Xm_ST(melt_wf,species):
    wm_ST = 100.*melt_wf['ST']
    return wm_ST/species.loc["S","M"]
def Xm_S(melt_wf,species):
    return wm_S(melt_wf)/species.loc["S","M"]
def Xm_SO3(melt_wf,species):
    return wm_SO3(melt_wf,species)/species.loc["SO3","M"]
def Xm_H2(melt_wf,species):
    wm_H2 = 100.*melt_wf['H2']
    return wm_H2/species.loc["H2","M"]

# Mole fraction in the melt based on mixing between volatile-free melt on a singular oxygen basis and volatiles
def Xm_m_so(run,melt_wf,setup,species): # singular oxygen basis
    return wm_nvol(melt_wf)/M_m_SO(run,setup,species)    
def Xm_tot_so(run,melt_wf,setup,species):
    return Xm_H2OT(melt_wf,species) + Xm_CO2(melt_wf,species) + Xm_m_so(run,melt_wf,setup,species) #+ Xm_S(wm_ST,species) + 
def xm_H2OT_so(run,melt_wf,setup,species):
    return Xm_H2OT(melt_wf,species)/Xm_tot_so(run,melt_wf,setup,species)
def xm_CO2_so(run,melt_wf,setup,species):
    return Xm_CO2(melt_wf,species)/Xm_tot_so(run,melt_wf,setup,species)
def xm_ST_so(run,melt_wf,setup,species):
    return Xm_ST(melt_wf,species)/Xm_tot_so(run,melt_wf,setup,species)
def xm_S_so(run,melt_wf,setup,species):
    return Xm_S(melt_wf,species)/Xm_tot_so(run,melt_wf,setup,species)
def xm_SO3_so(run,melt_wf,setup,species):
    return Xm_SO3(melt_wf,species)/Xm_tot_so(run,melt_wf,setup,species)
def xm_melt_so(run,melt_wf,setup,species):
    return Xm_m_so(run,melt_wf,setup,species)/Xm_tot_so(run,melt_wf,setup,species)
def Xm_t_so(run,melt_wf,setup,species):
    return xm_H2OT_so(run,melt_wf,setup,species)*species.loc["H2O","M"] + xm_CO2_so(run,melt_wf,setup,species)*species.loc["CO2","M"] + xm_melt_so(run,melt_wf,setup,species)*M_m_SO(run,setup,species)

# Mole fraction in the melt based on mixing between volatile-free melt on an oxide basis, H2O, and H2
def Xm_m_ox(run,melt_wf,setup,species): # singular oxygen basis
    return wm_nvol(melt_wf)/M_m_ox(run,setup,species)    
def Xm_tot_ox(run,melt_wf,setup,species):
    if melt_wf['H2'] > 0.:
        result = Xm_H2OT(melt_wf,species) + Xm_H2(melt_wf,species) + Xm_m_ox(run,melt_wf,setup,species) 
    else:
        result = Xm_H2OT(melt_wf,species) + Xm_m_ox(run,melt_wf,setup,species) 
    return result
def xm_H2OT_ox(run,melt_wf,setup,species):
    return Xm_H2OT(melt_wf,species)/Xm_tot_ox(run,melt_wf,setup,species)
def xm_H2_ox(run,melt_wf,setup,species):
    return Xm_H2(melt_wf,species)/Xm_tot_ox(run,melt_wf,setup,species)
def xm_melt_ox(run,melt_wf,setup,species):
    return Xm_m_ox(run,melt_wf,setup,species)/Xm_tot_ox(run,melt_wf,setup,species)
def Xm_t_ox(run,melt_wf,setup,species):
    if melt_wf["H2"] > 0.:
        result = xm_H2OT_ox(run,melt_wf,setup,species)*species.loc["H2O","M"] + xm_H2_ox(run,melt_wf,setup,species)*species.loc["H2","M"] + xm_melt_ox(run,melt_wf,setup,species)*M_m_ox(run,setup,species)
    else: result = xm_H2OT_ox(run,melt_wf,setup,species)*species.loc["H2O","M"] + xm_melt_ox(run,melt_wf,setup,species)*M_m_ox(run,setup,species)
    return result


##########################
### sulphur speciation ###
##########################

def S6S2(run,PT,melt_wf,setup,species,models):
    T_K = PT['T']+273.15
    model = models.loc["sulphate","option"]
    if model == "Nash19": # Nash, W.M., Smythe, D.J., Wood, B.J. (2019). Compositional and temperature effects on sulfur speciation and solubility in silicate melts. Earth and Planetary Science Letters 507:187-198. https://doi.org/10.1016/j.epsl.2018.12.006
        return pow(10.0,(8.0*math.log10(Fe3Fe2(melt_wf)) + ((8.7436e6)/pow(T_K,2.0)) - (27703.0/T_K) + 20.273))
    else:
        return (C_SO4(run,PT,melt_wf,setup,species,models)/C_S(run,PT,melt_wf,setup,species,models))*pow(f_O2(run,PT,melt_wf,setup,species,models),2.0)
    
def S6S2_2_fO2(S62,run,PT,setup,species,models):
    fO2 = ((S62*C_S(run,PT,melt_wf,setup,species,models))/C_SO4(run,PT,melt_wf,setup,species,models))**0.5
    return fO2

def S6ST(run,PT,melt_wf,setup,species,models):
    S6S2_ = S6S2(run,PT,melt_wf,setup,species,models)
    return S6S2_/(S6S2_+1.0)

def wm_S(run,PT,melt_wf,setup,species,models):
    wm_ST = 100.*melt_wf['ST']
    S6ST_ = S6ST(run,PT,melt_wf,setup,species,models)
    return wm_ST*(1.0-S6ST_)

def wm_SO3(run,PT,melt_wf,setup,species,models):
    wm_ST = 100.*melt_wf['ST']
    S6ST_ = S6ST(run,PT,melt_wf,setup,species,models)    
    return ((wm_ST*S6ST_)/species.loc["S","M"])*species.loc["SO3","M"]

def ratio2overtotal(x):
    return x/x+1.

def overtotal2ratio(x):
    return x/(1.-x)


##########################
### sulphur saturation ###
##########################

# sulphate content at anhydrite saturation (S6+CSS)
# Chowdhury, P. and Dasgupta, R. 2019. Effect of sulfate on the basaltic liquidus and Sulfur Concentration at Anhydrite Saturation (SCAS) of hydrous basalts – Implications for sulfur cycle in subduction zones. Chemical Geology, 522, 162–174, https://doi.org/10.1016/J.CHEMGEO.2019.05.020.
def SCAS(run,PT,melt_wf,setup,species): # sulphate content (ppm) at anhydrite saturation
    
    T = PT['T'] +273.15 # T in K
    wm_H2OT = 0.
    
    Wm_tot = setup.loc[run,"SiO2"] + setup.loc[run,"TiO2"] + setup.loc[run,"Al2O3"] + Wm_FeOT(run,setup,species) + setup.loc[run,"MnO"] + setup.loc[run,"MgO"] + setup.loc[run,"MnO"] + setup.loc[run,"CaO"] + setup.loc[run,"Na2O"] + setup.loc[run,"K2O"]
    mol_tot = ((100.0-wm_H2OT)*(((setup.loc[run,"SiO2"]/species.loc["SiO2","M"]) + (setup.loc[run,"TiO2"]/species.loc["TiO2","M"]) + (setup.loc[run,"Al2O3"]/species.loc["Al2O3","M"]) + (Wm_FeOT(run,setup,species)/species.loc["FeO","M"]) + (setup.loc[run,"MnO"]/species.loc["MnO","M"]) + (setup.loc[run,"MgO"]/species.loc["MgO","M"]) + (setup.loc[run,"CaO"]/species.loc["CaO","M"]) + (setup.loc[run,"Na2O"]/species.loc["Na2O","M"]) + (setup.loc[run,"K2O"]/species.loc["K2O","M"]))/Wm_tot)) + wm_H2OT/species.loc["H2O","M"]
    Si = (((100.0-wm_H2OT)*(setup.loc[run,"SiO2"]/Wm_tot))/species.loc["SiO2","M"])/mol_tot
    Ca = (((100.0-wm_H2OT)*(setup.loc[run,"CaO"]/Wm_tot))/species.loc["CaO","M"])/mol_tot
    Mg = (((100.0-wm_H2OT)*(setup.loc[run,"MgO"]/Wm_tot))/species.loc["MgO","M"])/mol_tot                                                                                            
    Fe = (((100.0-wm_H2OT)*(Wm_FeOT(run,setup,species)/Wm_tot))/species.loc["FeO","M"])/mol_tot
    Al = (((100.0-wm_H2OT)*(setup.loc[run,"Al2O3"]/Wm_tot))/species.loc["Al2O3","M"])/mol_tot
    Na = (((100.0-wm_H2OT)*(setup.loc[run,"Na2O"]/Wm_tot))/species.loc["Na2O","M"])/mol_tot
    K = (((100.0-wm_H2OT)*(setup.loc[run,"K2O"]/Wm_tot))/species.loc["K2O","M"])/mol_tot                                                                                            
    a = -13.23
    b = -0.50
    dSi = 3.02
    dCa = 36.70
    dMg = 2.84
    dFe = 10.14
    dAl = 44.28
    dNa = 26.27
    dK = -25.77
    e = 0.09
    f = 0.54
    dX = dSi*Si + dCa*Ca + dMg*Mg + dFe*Fe + dAl*Al + dNa*Na + dK*K
    lnxm_SO4 = a + b*((10.0**4.0)/T) + dX + e*wm_H2OT - f*gp.log(Ca)                                                                                  
    xm_SO4 = gp.exp(lnxm_SO4) 
    Xm_SO4 = xm_SO4*(xm_SO4 + mol_tot) 
    return Xm_SO4*species.loc["S","M"]*10000.0

# sulphide content at sulphide saturation (S2-CSS)
# O'Neill, H.S. (2021). The Thermodynamic Controls on Sulfide Saturation in Silicate Melts with Application to Ocean Floor Basalts. In Magma Redox Geochemistry (eds R. Moretti and D.R. Neuville). https://doi.org/10.1002/9781119473206.ch10
def SCSS(run,PT,melt_wf,setup,species,models): # sulphide content (ppm) at sulphide saturation 
    
    model = models.loc["sulphide","option"]
    P_bar = PT['P'] # P in bar
    T = PT['T'] + 273.15 # T in K
    Fe3FeT = melt_wf["Fe3FeT"]
    
    # Mole fractions in the melt on cationic lattice (all Fe as FeO) no volatiles
    tot = ((species.loc["SiO2","no_cat"]*setup.loc[run,"SiO2"])/species.loc["SiO2","M"]) + ((species.loc["TiO2","no_cat"]*setup.loc[run,"TiO2"])/species.loc["TiO2","M"]) + ((species.loc["Al2O3","no_cat"]*setup.loc[run,"Al2O3"])/species.loc["Al2O3","M"]) + ((species.loc["FeO","no_cat"]*Wm_FeOT(run,setup,species))/species.loc["FeO","M"]) + ((species.loc["MgO","no_cat"]*setup.loc[run,"MgO"])/species.loc["MgO","M"]) + ((species.loc["MnO","no_cat"]*setup.loc[run,"MnO"])/species.loc["MnO","M"]) + ((species.loc["CaO","no_cat"]*setup.loc[run,"CaO"])/species.loc["CaO","M"]) + ((species.loc["Na2O","no_cat"]*setup.loc[run,"Na2O"])/species.loc["Na2O","M"]) + ((species.loc["K2O","no_cat"]*setup.loc[run,"K2O"])/species.loc["K2O","M"]) + ((species.loc["P2O5","no_cat"]*setup.loc[run,"P2O5"])/species.loc["P2O5","M"]) 
    Si = ((species.loc["SiO2","no_cat"]*setup.loc[run,"SiO2"])/species.loc["SiO2","M"])/tot
    Ti = ((species.loc["TiO2","no_cat"]*setup.loc[run,"TiO2"])/species.loc["TiO2","M"])/tot
    Al = ((species.loc["Al2O3","no_cat"]*setup.loc[run,"Al2O3"])/species.loc["Al2O3","M"])/tot
    Fe = ((species.loc["FeO","no_cat"]*Wm_FeOT(run,setup,species))/species.loc["FeO","M"])/tot
    Mn = ((species.loc["MnO","no_cat"]*setup.loc[run,"MnO"])/species.loc["MnO","M"])/tot
    Mg = ((species.loc["MgO","no_cat"]*setup.loc[run,"MgO"])/species.loc["MgO","M"])/tot
    Ca = ((species.loc["CaO","no_cat"]*setup.loc[run,"CaO"])/species.loc["CaO","M"])/tot
    Na = ((species.loc["Na2O","no_cat"]*setup.loc[run,"Na2O"])/species.loc["Na2O","M"])/tot
    K = ((species.loc["K2O","no_cat"]*setup.loc[run,"K2O"])/species.loc["K2O","M"])/tot
    Fe2 = Fe*(1-Fe3FeT)
    
    R = 8.31441
    P = (1.0e-4)*P_bar # pressure in GPa
    D = (137778.0 - 91.66*T + 8.474*T*gp.log(T)) # J/mol
    sulphide_comp = 1.0 # assumes the sulphide is pure FeS (no Ni, Cu, etc.)
        
    lnaFeS = gp.log((1.0 - Fe2)*sulphide_comp)
    lnyFe2 = (((1.0-Fe2)**2.0)*(28870.0 - 14710.0*Mg + 1960.0*Ca + 43300.0*Na + 95380.0*K - 76880.0*Ti) + (1.0-Fe2)*(-62190.0*Si + 31520.0*Si**2.0))/(R*T)
    lnS = D/(R*T) + gp.log(C_S(run,PT,melt_wf,setup,species,models)) - gp.log(Fe2) - lnyFe2 + lnaFeS + (-291.0*P + 351.0*gp.erf(P))/T
    return gp.exp(lnS)


#######################
### iron speciation ###
#######################

def Fe3Fe2(melt_wf):
    Fe3FeT = melt_wf['Fe3FeT']
    return Fe3FeT/(1.0 - Fe3FeT)

def Wm_FeT(run,setup,species):
    if setup.loc[run,"FeOT"] > 0.0:
        return (setup.loc[run,"FeOT"]/species.loc["FeO","M"])*species.loc["Fe","M"]
    elif setup.loc[run,"Fe2O3T"] > 0.0:
        return (setup.loc[run,"Fe2O3T"]/species.loc["Fe2O3","M"])*species.loc["Fe","M"]
    else:
        return ((setup.loc[run,"FeO"]/species.loc["FeO","M"]) + (setup.loc[run,"Fe2O3"]/species.loc["Fe2O3","M"]))*species.loc["Fe","M"]

def Wm_FeO(run,melt_wf,setup,species):
    Fe3FeT = melt_wf['Fe3FeT']
    return (Wm_FeT(run,setup,species)/species.loc["Fe","M"])*(1.0-Fe3FeT)*species.loc["FeO","M"]

def Wm_Fe2O3(run,melt_wf,setup,species):
    Fe3FeT = melt_wf['Fe3FeT']
    return (Wm_FeT(run,setup,species)/species.loc["Fe","M"])*Fe3FeT*species.loc["Fe2O3","M"]

def Wm_FeOT(run,setup,species):
    return (Wm_FeT(run,setup,species)/species.loc["Fe","M"])*species.loc["FeO","M"]

def wm_Fe_nv(run,melt_wf,setup,species): # no volatiles
    Wm_tot = setup.loc[run,"SiO2"] + setup.loc[run,"TiO2"] + setup.loc[run,"Al2O3"] + setup.loc[run,"MnO"] + setup.loc[run,"MgO"] + setup.loc[run,"MnO"] + setup.loc[run,"CaO"] + setup.loc[run,"Na2O"] + setup.loc[run,"K2O"] + setup.loc[run,"P2O5"] + Wm_FeO(run,melt_wf,setup,species) + Wm_Fe2O3(run,melt_wf,setup,species)
    FeT = species.loc["Fe","M"]*((2.0*Wm_Fe2O3(run,melt_wf,setup,species)/species.loc["Fe2O3","M"]) + (Wm_FeO(run,melt_wf,setup,species)/species.loc["FeO","M"]))
    return 100.0*FeT/Wm_tot

def Fe3FeT_i(run,PT,setup,species,models):
    model = models.loc["fO2","option"]
    T_K = PT['T']+273.15
    
    if model == "buffered":
        fO2 = 10**(setup.loc[run,"logfO2"])
        return fO22Fe3FeT(fO2,run,PT,setup,species,models)
    else:
        if pd.isnull(setup.loc[run,"Fe3FeT"]) == False:
            return setup.loc[run,"Fe3FeT"]
        elif pd.isnull(setup.loc[run,"logfO2"]) == False:
            fO2 = 10.0**(setup.loc[run,"logfO2"])
            return fO22Fe3FeT(fO2,run,PT,setup,species,models)
        elif pd.isnull(setup.loc[run,"DNNO"]) == False:
            D = setup.loc[run,"DNNO"]
            fO2 = Dbuffer2fO2(PT,D,"NNO")
            return fO22Fe3FeT(fO2,run,PT,setup,species,models)
        elif pd.isnull(setup.loc[run,"DFMQ"]) == False:
            D = setup.loc[run,"DFMQ"]
            fO2 = Dbuffer2fO2(PT,D,"FMQ")
            return fO22Fe3FeT(fO2,run,PT,setup,species,models)
        else:
            return ((2.0*setup.loc[run,"Fe2O3"])/species.loc["Fe2O3","M"])/(((2.0*setup.loc[run,"Fe2O3"])/species.loc["Fe2O3","M"]) + (setup.loc[run,"FeO"]/species.loc["FeO","M"]))
           
def fO22Fe3FeT(fO2,run,PT,setup,species,models):
    model = models.loc["fO2","option"]
    T_K = PT['T']+273.15
    
    if model == "Kress91":
        a = 0.196
        PTterm = PT_KCterm(PT)
        lnXFe2O3XFeO = a*gp.log(fO2) + PTterm
        XFe2O3XFeO = gp.exp(lnXFe2O3XFeO)
        return (2.0*XFe2O3XFeO)/((2.0*XFe2O3XFeO)+1.0)
    
    elif model == "Kress91A": 
        KD2 = 0.4
        y = 0.3
        kd1 = KD1(run,PT,setup,species,models)
        XFeO15XFeO = ((kd1*fO2**0.25)+(2.0*y*KD2*(kd1**(2.0*y))*(fO2**(0.5*y))))/(1.0 + (1.0 - 2.0*y)*KD2*(kd1**(2.0*y))*(fO2**(0.5*y)))
        return XFeO15XFeO/(XFeO15XFeO+1.0)  
    
    elif model == "ONeill18": # O'Neill et al. (2018) EPSL 504:152-162
        # mole fractions on a single cation basis
        tot = ((species.loc["SiO2","no_cat"]*setup.loc[run,"SiO2"])/species.loc["SiO2","M"]) + ((species.loc["TiO2","no_cat"]*setup.loc[run,"TiO2"])/species.loc["TiO2","M"]) + ((species.loc["Al2O3","no_cat"]*setup.loc[run,"Al2O3"])/species.loc["Al2O3","M"]) + ((species.loc["FeO","no_cat"]*Wm_FeOT(run,setup,species))/species.loc["FeO","M"]) + ((species.loc["MgO","no_cat"]*setup.loc[run,"MgO"])/species.loc["MgO","M"]) + ((species.loc["MnO","no_cat"]*setup.loc[run,"MnO"])/species.loc["MnO","M"]) + ((species.loc["CaO","no_cat"]*setup.loc[run,"CaO"])/species.loc["CaO","M"]) + ((species.loc["Na2O","no_cat"]*setup.loc[run,"Na2O"])/species.loc["Na2O","M"]) + ((species.loc["K2O","no_cat"]*setup.loc[run,"K2O"])/species.loc["K2O","M"]) + ((species.loc["P2O5","no_cat"]*setup.loc[run,"P2O5"])/species.loc["P2O5","M"])
        Ca = ((species.loc["CaO","no_cat"]*setup.loc[run,"CaO"])/species.loc["CaO","M"])/tot
        Na = ((species.loc["Na2O","no_cat"]*setup.loc[run,"Na2O"])/species.loc["Na2O","M"])/tot
        K = ((species.loc["K2O","no_cat"]*setup.loc[run,"K2O"])/species.loc["K2O","M"])/tot
        P = ((species.loc["P2O5","no_cat"]*setup.loc[run,"P2O5"])/species.loc["P2O5","M"])/tot
        DQFM = gp.log10(fO2) - (8.58 - (25050/T_K)) # O'Neill (1987)
        lnFe3Fe2 = 0.25*DQFM - 1.36 + 2.4*Ca + 2.0*Na + 3.7*K - 2.4*P
        Fe3Fe2 =  gp.exp(lnFe3Fe2)
        return Fe3Fe2/(Fe3Fe2 + 1.0)


###################################
### concentration of insolubles ### 
###################################

def conc_insolubles(run,PT,melt_wf,setup,species,models):
    S2m = melt_wf["S2-"] # weight fraction of S2-
    S6p = (C_SO4(run,PT,melt_wf,setup,species,models)*f_O2(run,PT,melt_wf,setup,species,models)**2*S2m)/C_S(run,PT,melt_wf,setup,species,models) # weight fraction S6+
    S_T = S2m + S6p
    S2m_ST = S2m/S_T
    S6p_ST = S6p/S_T
    return S6p, S2m_ST, S6p_ST, S_T

    
############################################
### converting gas and melt compositions ###
############################################

# calculate weight fraction of species in the gas
def gas_wf(gas_mf,species):
    wg_O2 = (species.loc["O2","M"]*gas_mf["O2"])/gas_mf["Xg_t"]
    wg_S2 = (species.loc["S2","M"]*gas_mf["S2"])/gas_mf["Xg_t"]
    wg_SO2 = (species.loc["SO2","M"]*gas_mf["SO2"])/gas_mf["Xg_t"]
    return wg_O2, wg_S2, wg_SO2

# calculate weight fraction of species in the gas relative to total system
def gas_wft(gas_mf,species):
    wg_O2, wg_S2, wg_SO2 = gas_wf(gas_mf,species)
    wt_g = gas_mf['wt_g']
    wgt_O2 = wg_O2*wt_g
    wgt_S2 = wg_S2*wt_g
    wgt_SO2 = wg_SO2*wt_g
    return wgt_O2, wgt_S2, wgt_SO2

def gas_weight(gas_mf,bulk_wf,species):
    wgt_O2, wgt_S2, wgt_SO2 = gas_wft(gas_mf,species)
    Wg_O2 = wgt_O2*bulk_wf['Wt']
    Wg_S2 = wgt_S2*bulk_wf['Wt']
    Wg_SO2 = wgt_SO2*bulk_wf['Wt']
    Wg_t = gas_mf['wt_g']*bulk_wf['Wt']
    return Wg_O2, Wg_S2, Wg_SO2, Wg_t

def gas_moles(gas_mf,bulk_wf,species):
    Wg_O2, Wg_S2, Wg_SO2, Wg_t = gas_weight(gas_mf,bulk_wf,species)
    Xg_O2 = Wg_O2/species.loc["O2","M"]
    Xg_S2 = Wg_S2/species.loc["S2","M"]
    Xg_SO2 = Wg_SO2/species.loc["SO2","M"]
    Xt_g = Xg_O2 + Xg_S2 + Xg_SO2
    return Xg_O2, Xg_S2, Xg_SO2, Xt_g
        
# calculate weight fraction of elements in the gas
def gas_elements(gas_mf,species):
    wg_O2, wg_S2, wg_SO2 = gas_wf(gas_mf,species)
    wg_O = wg_O2 + species.loc["O","M"]*((2.*wg_SO2/species.loc["SO2","M"]))
    wg_S = wg_S2 + species.loc["S","M"]*((wg_SO2/species.loc["SO2","M"]))
    return wg_O, wg_S

# calculate weight fraction of elements in the melt
def melt_elements(run,PT,melt_wf,bulk_wf,gas_comp,setup,species,models):
    wm_S = melt_wf["ST"]
    wm_Fe = bulk_wf["Fe"]/(1.-gas_comp["wt_g"])
    S6T = S6ST(run,PT,melt_wf,setup,species,models)
    wm_O = species.loc['O','M']*(((wm_Fe*(1.5*melt_wf["Fe3FeT"]+(1.-melt_wf["Fe3FeT"])))/species.loc['Fe','M'])+((wm_S*3.*S6T)/species.loc['S','M']))
    return wm_S, wm_Fe, wm_O