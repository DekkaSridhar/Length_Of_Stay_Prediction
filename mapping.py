import pandas as pd
df=pd.read_csv("Hospital_Inpatient_Discharges__SPARCS_De-Identified___2015.csv")
ccsd=df["CCS Diagnosis Description"].unique()
ccsc={}
for i in  ccsd:
    ccsc[i]=df.loc[df['CCS Diagnosis Description']==i,'CCS Diagnosis Code'].unique()[0]
aprd=df["APR MDC Description"].unique()
aprc={}
for i in  aprd:
    aprc[i]=df.loc[df['APR MDC Description']==i,'APR MDC Code'].unique()[0]
aprdrgd=df["APR DRG Description"].unique()
aprdrgc={}
for i in  aprdrgd:
    aprdrgc[i]=df.loc[df['APR DRG Description']==i,'APR DRG Code'].unique()[0]
print(aprc)