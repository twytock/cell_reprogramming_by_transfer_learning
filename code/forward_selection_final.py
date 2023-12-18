"""
forward_selection_final.py

Created by Thomas Wytock on 2023-11-06.
"""

import sys, os, os.path as osp
import numpy as np
import pandas as pd
import time
from scoop import futures
from sklearn.neighbors import KNeighborsClassifier as KNC
from functools import partial
from scipy.stats import percentileofscore as POS
import cplex
from operator import itemgetter as ig
from glob import glob


ctf = sys.argv[1]; dmode = sys.argv[2]; CH = sys.argv[3]; 
## dmode should be one of GeneExp or RNASeq
IN = 'data'
OUT = 'output'

n_neighbors=8

## set filenames, start from the dimension-reduced deltas

if dmode=='RNASeq':
    DATA_FN = 'proj_data_df.csv' 
    DELTA_FN = 'proj_perturbations_df.csv' ## 28M file, OK!
    FEAT_FN = 'GTEx_feat_l_bc.txt'
    DF = pd.read_table(f'{IN}/{dmode}/gene_symbol_mappings.txt.gz',sep='\t',compression='gzip',header=0)
    GENE2ENS = DF.set_index('Gene name')['Gene stable ID version']
    ENS2ENS = DF.set_index('Gene stable ID')['Gene stable ID version']
    ENS2GENE = DF.set_index('Gene stable ID version')['Gene name']
    REF2ENS = DF.set_index('RefSeq match transcript')['Gene stable ID version']
    G2ED = GENE2ENS.to_dict()
    E2GD = ENS2GENE.to_dict()
    E2ED = ENS2ENS.to_dict()
else:
    ## dmode=='GeneExp'
    DATA_FN = 'nonseq_bc_corr_data_all.csv'
    #EVEC_FN = 'pc_pearson_nonseq_evec.pkl' ## 1.2G file,
    DELTA_FN = 'delta_corr_newnci60.csv.gz' ## can 'delta_corr_newnci60-2.pkl' be used?
    FEAT_FN = 'tissue_feat_l_WC_all.txt'
    ps_gs_map = pd.read_csv(f'{IN}/{dmode}/probe_gsym_mapping.csv',index_col=0).iloc[:,0]

with open(f'{IN}/{dmode}/{FEAT_FN}','r') as fh:
    feat_l = [ln.strip() for ln in fh]
    if dmode=='GeneExp':
        feat_l = feat_l[:4]
def weights(df):
    WT = df.std(axis=0,ddof=1).apply(lambda i: 1/i)
    if np.any(np.isinf(WT)):
        M = np.amax(WT[~np.isinf(WT)])
        WT[np.isinf(WT)]=2*M
    WT = WT.div(np.linalg.norm(WT.astype(np.float64)))
    return WT

def convert_ens_to_ens2(INDS):
    '''
    Function to convert the Ensembl gene identifiers so they agree with the perturbations
    Parameters
    ----------
    INDS : list or pandas Multiindex
        ensembl, gene symbol are converted to be in common.

    Returns
    -------
    pandas MultiIndex
        Multiindex with common Ensembl Ids based on gene symbols.
    '''
    # LLL is a list of ensemblIds
    LLL = []
    for ens,gene in INDS:
        ## check both the Ensembl gene ID and Gene symbol
        E1 = E2ED.get(ens.split('.')[0],'')
        E2 = G2ED.get(gene,'')
        if len(E1)>0 and len(E2)>0:
            if E1!=E2:
                # if mismatch, revert to the ensemblId associated with symbol
                LLL.append(E2)
            else:
                LLL.append(E1)
        elif len(E1)>0:
            LLL.append(E1)
        elif len(E2)>0:
            LLL.append(E2) 
        else:
            LLL.append('')
    # recover the associated gene symbols for multiindex
    GNS = [E2GD.get(l,'') for l in LLL]
    return pd.MultiIndex.from_tuples(zip(LLL,GNS))

def build_knn_model(proj_data,LBLS,n_neighbors=8):
    '''
    builds k-nearest neighbors model from training data

    Parameters
    ----------
    proj_data : pandas DataFrame
        transcriptomic data (projected onto eigengenes)
    LBLS : pandas Categorical
        list of labels (e.g. cell types)
    n_neighbors : integer, optional
        the number of neighbors to include. The default is 8.

    Returns
    -------
    clf : sci-kit learn KNeighborsClassifier object, 
        The k-nearest neighbors classifier object, fit to training data.
    '''
    clf = KNC(n_neighbors,'distance')
    clf.fit(proj_data.T.values,LBLS.codes)
    return clf

def cplex_optimize_const(Q,cvec,**kwargs):
    '''
    Perform constrained optimization with CPLEX

    Parameters
    ----------
    Q : 2-D, square, numpy array
        Quadratic component of the objective function
    cvec : numpy array
        Linear component of the objective function; same length as Q
    kwargs : dict of optional arguments
        lb : lower bounds for the variables
        ub : upper bounds for the variables
        
    Returns
    -------
    x : list
        optimal values of the control inputs
    lpstat : int
        status of the cplex solver

    '''
    Q = 0.5*(Q+Q.T)
    Q = np.atleast_2d(Q) if not isinstance(Q,np.ndarray) else Q
    numvars = Q.shape[0];
    Q = Q + np.eye(numvars)*1e-12
    cvec = cvec[:,0]
    qmat = []
    for ii in range(numvars):
        qmat.append([range(numvars),Q[ii,:].tolist()])
    c = cplex.Cplex()
    out = c.set_results_stream(None)
    out = c.set_log_stream(None)
    out = c.set_error_stream(None)
    out = c.set_warning_stream(None)
    c.parameters.optimalitytarget.set(c.parameters.optimalitytarget.values.optimal_global)
    if 'lb' in kwargs.keys() and 'ub' in kwargs.keys():
        LB = kwargs['lb']
        LB = LB if isinstance(LB,np.ndarray) else np.ones(numvars)*LB
        LB = LB.tolist()
        UB = kwargs['ub']
        UB = UB if isinstance(UB,np.ndarray) else np.ones(numvars)*UB
        UB = UB.tolist()
        c.variables.add(obj=cvec.tolist(),ub=UB,lb=LB)
    elif 'lb' in kwargs.keys():
        LB = kwargs['lb']
        LB = LB if isinstance(LB,np.array) else np.ones(numvars)*LB
        LB = LB.tolist()
        c.variables.add(obj=cvec.tolist(),lb=LB)
    elif 'ub' in kwargs.keys():
        UB = kwargs['ub']
        UB = UB if isinstance(UB,np.array) else np.ones(numvars)*UB
        UB = UB.tolist()
        c.variables.add(obj=cvec.tolist(),ub=UB)
    else:
        c.variables.add(obj=cvec.tolist())        
    c.objective.set_sense(c.objective.sense.minimize)
    c.objective.set_quadratic(qmat)
    c.solve()
    lpstat = c.solution.get_status()
    x = c.solution.get_values()
    return x,lpstat
    
def updated_recw_cplex_fs(kw,dcols,S,T,deltas,W,lb,ub,opt):
    '''
    Convert initial states, selected columns, and different optimization parameters
    into the inputs appropriate for calling Cplex.
    
    Parameters
    ----------
    kw : name of the column to add
    dcols : list or pandas Index, names of the perturbations already selected
    S : pandas Series, initial state gene expression vector
    T : pandas Series, target state gene expression vector
    deltas : pandas DataFrame, observed transcriptional responses of the perturbations
    W : pandas Series, weights of each eigengene
    lb : float, lower bounds for the variables
    ub : float, upper bounds for the variables
    opt : Boolean, whether to calculate the optimal solution (True) or naive solution (False)
    
    Returns
    -------
    kw : label of the perturbation
    al_df : pandas DataFrame, contains the amounts that each gene needs to be perturbed
    d1 : pandas Series, final distance to the target for the optimal perturbation
    pf : pandas Series, percentage of distance to the target recovered for the optimal perturbation
    
    '''
    
    if not opt:
        if isinstance(dcols,list):
            dcols = dcols + [kw]
        else:
            dcols = dcols.tolist()+ [kw] # add kw to index
    d = deltas.loc[:,dcols]
    if np.any(np.isinf(W)):
        M=np.amax(W[~np.isinf(W)])
    wsep = (S.iloc[0].sub(T.iloc[0]).multiply(W)).to_frame()
    d0 = np.linalg.norm(wsep)
    if isinstance(d,pd.DataFrame):
        wD = d.T.multiply(W).T
        wC = wsep.T.dot(wD).T
        wQ = wD.T.dot(wD)
    elif isinstance(d,pd.Series):
        wD = d.T.multiply(W).T
        wC = wsep.dot(wD)
        wQ = wD.T.dot(wD)
    al,sc = cplex_optimize_const(wQ.values,wC.values,lb=lb,ub=ub)
    if isinstance(d,pd.DataFrame):
        d1 = np.linalg.norm(wsep.add(wD.dot(pd.DataFrame(al,index=wQ.index))))
    elif isinstance(d,pd.Series):
        d1 = np.linalg.norm(wsep.add(wD.dot(pd.Series(al,index=wD.name))))
    pf = 1-d1/d0
    return kw,np.atleast_2d(al),d1,pf

def updated_recw_calc_fs(kw,dcols,S,T,deltas,W,opt):
    '''
    Convert initial states, selected columns, and different optimization parameters
    into a linear algebra equation for finding the optimal perturbation.
    
    Parameters
    ----------
    kw : name of the column to add
    dcols : list or pandas Index, names of the perturbations already selected
    S : pandas Series, initial state gene expression vector
    T : pandas Series, target state gene expression vector
    deltas : pandas DataFrame, observed transcriptional responses of the perturbations
    W : pandas Series, weights of each eigengene
    lb : float, lower bounds for the variables
    ub : float, upper bounds for the variables
    opt : Boolean, whether to calculate the optimal solution (True) or naive solution (False)
    
    Returns
    -------
    kw : label of the perturbation
    al : pandas DataFrame, contains the amounts that each gene needs to be perturbed
    d1 : pandas Series, final distance to the target for the optimal perturbation
    pf : pandas Series, percentage of distance to the target recovered for the optimal perturbation
    
    '''
    if not opt:
        if isinstance(dcols,list):
            dcols = dcols + [kw]
        else:
            dcols = dcols.tolist()+ [kw] # add kw to index
    d = deltas.loc[:,dcols] 
    wsep = (S.iloc[0].sub(T.iloc[0]).multiply(W)).to_frame()
    if isinstance(d,pd.DataFrame):
        wD = d.T.multiply(W).T
        wC = wsep.T.dot(wD).T
        wQ = wD.T.dot(wD)
        al = -1*np.linalg.solve(wQ.values,wC.loc[wQ.index].values)
        #d1 = np.linalg.norm(WSEP+np.dot(WD,al))
        d1 = np.linalg.norm(wsep.add(wD.dot(pd.DataFrame(al,index=wQ.index))))
    else:
        wD = d.multiply(W)
        wC = wsep.dot(wD); wQ = wD.dot(wD)
        al = np.asarray([-1 * wC.values/wQ.values])
        d1 = np.linalg.norm(wsep.add(wD.dot(pd.Series(al,index=wQ.index))))
    d0 = np.linalg.norm(wsep)
    pf = 1-d1/d0
    return kw,al,d1,pf

def updated_add_1_col(selcols,init,targ,deltas,W,opt=False):
    '''
    Add the single gene that most improves reprogramming to the perturbation set 

    Parameters
    ----------
    selcols : list
        list of column identifiers for the genetic perturbations.
    init : pandas Series
        expression values of the initial state.
    targ : pandas Series
        expression values of the final state.
    deltas : pandas DataFrame
        perturbation matrix; rows = gene responses, columns = perturbations.
    W : pandas Series
        weights vector to determine how each feature applies to data.
    opt : boolean, optional
        whether or not to calculate the optimal perturbation. The default is False.
        if opt is True, only the optimal perturbation including all genes will be performed
        otherwise, forward selection will proceed fixing the columns in "selcols" and adding
        columns one at a time
    CH : string (global variable)
        character that determines which constraint class to use.
        A -- unconstrained
        I -- u \in [-1, 1]
        E -- u \in [ 0, 1]

    Returns
    -------
    results : list
        list of distance recoveries and the associated perturbations.

    '''
    
    #uses W(weights) instead of tsigs(variances) (more convenient for this application)
    #init and targ should be UNWEIGHTED inputs
    addcols = list(set(deltas.columns)-set(selcols))
    if CH in ['I','E']:
        #print "CPLEX selected"
        if CH=='E':
            p_opt= partial(updated_recw_cplex_fs,dcols=selcols,S=init,T=targ,
            deltas=deltas,W=W,lb=0,ub=1,opt=False)
        elif CH=='I':
            p_opt= partial(updated_recw_cplex_fs,dcols=selcols,S=init,T=targ,
            deltas=deltas,W=W,lb=-1,ub=1,opt=False)
    elif CH in ['A']:
        #print "Linear Algebra selected"
        p_opt=partial(updated_recw_calc_fs,dcols=selcols,S=init,T=targ,deltas=deltas,W=W,opt=False)
    results = map(p_opt,addcols)
    #results = list(futures.map(p_opt,addcols))
    return results

def updated_reduce_fs(res,dcols,opt,Swt,dwt,model):
    '''
    Gather forward selection data into a vector of results and the oselected columns

    Parameters
    ----------
    res : tuple of 2 elements
        column (1st element) and amount of optimal perturbations (2nd element).
    dcols : list
        selected perturbation columns.
    opt : Boolean
        whether or not this is an optimal perturbation including all columns.
    Swt : pandas Series
        weighted intial state.
    dwt : pandas DataFrame
        DESCRIPTION.
    model : sci-kit learn KNN classifier
        KNN model that predicts cell type based on gene expression.

    Returns
    -------
    tuple
        dat_ser -- the optimal perturbation, distance recovery, and reprogramming statistics
        sc -- the selected columns.

    '''
    # instead of re-fitting the model, pass it as a parameter
    # S is simply the initial state from the scaled data
    # deltas are the perturbations scaled by the weights
    al_ser= pd.Series(np.zeros(dwt.shape[1]),index=dwt.columns)
    al_ser = al_ser.sort_index()
    if not opt:
        if isinstance(dcols,list):
            sc = dcols+[res[0]]
        else:
            sc = dcols.tolist() + [res[0]] # add column
    else:
        sc = dcols
    if len(sc)>0:
        al_ser.loc[sc]=res[1].ravel()
        d = dwt.loc[:,sc]
        R = Swt.add(d.dot(al_ser.loc[sc]))
    else:
        R=Swt
    C = model.predict(R)[0]
    P = model.predict_proba(R)[0,C]
    dat_ser=pd.Series(list(res[2:])+[C,P],index=['D_F','P_F','KNN_GP','KNN_PRB'])
    dat_ser = pd.concat([dat_ser,al_ser])
    return dat_ser,sc


def forward_selection(S,T,sel_cols,deltas,model,WT,targ_basin,IM=False):
    """
    Iteratively calculate the optimal perturbations for a given pair of initial and target states.
    
    
    Parameters
    ----------
    S : tuple, contains the initial state identifier as well as a pandas Series of the state gene expression profile
    T : pandas Series, the target state gene expression profile
    sel_cols : list, columns selected to optimize the state
    deltas : pandas DataFrame, contains the transcriptional responses of the perturbations to be optimized
    model : sklearn KNN model, function that maps the transcriptional state to a probability of belonging to each cell type
    WT : pandas Series, weights of each eigengene in determining the cell type
    targ_basin : int, integer classification label associated with the target cell type in the KNN model
    IM : Boolean (optional), whether to save the intermediate results to a temporary directory
    
    Returns
    -------
    S.index[0] : string, label of the initial state
    DF : pandas DataFrame, contains the statistics of the optimal perturbation
    """
    gsm,S = S
    S = S.to_frame().T
    p_add = partial(updated_add_1_col,init=S,targ=T,deltas=deltas,W=WT)
    p_red = partial(updated_reduce_fs,Swt=S.multiply(WT),dwt=deltas.T.multiply(WT).T,
                    model=model)
    red_ser_d = {}
    ## test for improper basins
    if not model.predict(T.multiply(WT))[0]==targ_basin:
        print("Invalid Target")
        #find point closest to the center to aim at
        T = proj_data.groupby(level=1).first().loc[ctf]
        p_add = partial(updated_add_1_col,init=S,targ=T,deltas=deltas,W=WT)
    if model.predict(S.multiply(WT))[0]==model.predict(T.multiply(WT))[0]:
        print("Overlap")
        res = ['none',
               pd.Series(np.zeros(deltas.shape[1]),index=deltas.columns),
               np.linalg.norm(S.sub(T).multiply(WT)),0]
        red_ser, sel_cols = p_red(res,[],True)
        red_ser_d[-1]=red_ser
        DF = pd.DataFrame(red_ser_d).T
        DF.index.name='NG'; DF=DF.reset_index()
        return S.name,DF
    if CH == 'E':
        opt_res = updated_recw_cplex_fs('opt_al',deltas.columns,S,T,deltas,WT,0,1,True)
        opt_red,opt_sc = updated_reduce_fs(opt_res,deltas.columns,True,S.multiply(WT),deltas.T.multiply(WT).T,model)   
        NGMAX = deltas.shape[1] - np.sum(opt_res[1]==0)
        NGrange = range(len(sel_cols),NGMAX)
    else: 
        NGrange = range(len(sel_cols),len(deltas.columns))
    for zz,NG in enumerate(NGrange):
        opt=True if len(sel_cols) == len(deltas.columns) else False
        res_l =p_add(sel_cols)
        res = sorted(res_l,key=ig(2))[0]
        red_ser, sel_cols = p_red(res,sel_cols,opt)
        red_ser_d[NG+1] = red_ser
        if red_ser.KNN_GP == targ_basin[0]:
            break
        if CH == 'E':
            red_ser_d[len(deltas.columns)] = opt_red
    DF = pd.DataFrame(red_ser_d).T
    DF.index.name='NG'; DF=DF.reset_index()
    if IM:
        cti = proj_data.xs(gsm,level=0).index[0]
        ctip = cti.replace(' - ','_')
        ctfp = ctf.replace(' - ','_')
        tmpdir = osp.join(OUT,f'FS_{dmode}/{ctip}-{ctfp}')
        if not osp.exists(tmpdir):
            os.mkdir(tmpdir)
        DF.to_pickle(osp.join(tmpdir,f'{S.index[0]}-{CH}.pkl'))
    return S.index[0],DF


def pandas_FS(cti,cti_df,targ):
    """
    Calculate the optimal perturbations for each initial state of a given cell type-target pair,
    and save the output as a file.
    
    Parameters
    ----------
    cti : string, initial cell type
    cti_df : pandas DataFrame, expression states associated with cell type `cti`
    targ : pandas Series, target expression state of cell type `ctf`
    
    Returns
    -------
    0
    """    
    print(f"Starting cell type pair {cti} -> {ctf}")
    ti = time.time()
    INTERMED = True if cti_df.shape[0]>50 else False
    part_FS = partial(forward_selection,T=targ,sel_cols=[],
                      deltas=proj_deltas,WT=WT,model=clf,
                      targ_basin=idx.get_indexer([ctf]),IM=INTERMED)
    ctip = cti.replace(' - ','_')
    ctfp = ctf.replace(' - ','_')
    if INTERMED:
        tmpdir= osp.join(OUT,dmode,f'{ctip}-{ctfp}')
        if not osp.exists(tmpdir):
            os.makedirs(tmpdir)
        comp_gr_df_l = []
    #gsm_res_df_l= map(part_FS,cti_df.iterrows())
    gsm_res_df_l= list(futures.map(part_FS,cti_df.iterrows()))
    if INTERMED:
        gsm_res_df_l.extend(comp_gr_df_l)
    df_l = []
    for gsm,df in gsm_res_df_l:
        df['GSM_i']=[gsm for _ in range(df.shape[0])]
        df_l.append(df)
    DF = pd.concat(df_l)
    DF['CTI']=[cti for _ in range(DF.shape[0])]
    DF['CTF']=[ctf for _ in range(DF.shape[0])]
    if not osp.exists(f'{OUT}/FS_{dmode}/{CH}'):
        os.makedirs(f'{OUT}/FS_{dmode}/{CH}')
    DF.to_pickle(f'{OUT}/FS_{dmode}/{CH}/{ctip}-{ctfp}.pkl')
    if INTERMED:
        print("Cleaning intermediate files.")
        map(os.remove,glob(osp.join(tmpdir,'*.pkl')))
    tf = time.time()
    TT = tf-ti
    print(f"T = {TT:.2f}\n")
    return 0

def FS():
    """
    Iterate over the initial cell types for a given final cell type and 
    calculate the optimal perturbations for each initial state.
    Parameters
    ----------
        
    Returns
    -------
    """
    ### need to get this and forward selection corrected
    GB = proj_data.xs(ctf,level=1)
    MU = GB.mean().to_frame().T
    ctfp = ctf.replace(' - ','_')
    for cti in cts:
        ctip = cti.replace(' - ','_')
        if cti == ctf: continue
        GL = f'{OUT}/FS_{dmode}/{CH}/{ctip}-{ctfp}.pkl'
        if osp.exists(GL):
            continue
        else:
            print(cti)
            cti_df = proj_data.xs(cti,level=1)
            z = pandas_FS(cti,cti_df,MU)
        print(f"{cti} calculated")
    return 0

def fix_indices(data):
    '''
    changes the cell type so all tissues in the human body index are included

    Parameters
    ----------
    mod_data : pandas DataFrame
        DataFrame of transcriptional data whose index will be modified
        
    Returns
    -------
    mod_data : pandas DataFrame
        DataFrame with modified cell types in level 1 of the MultiIndex

    '''
    L = []
    ## loop over indicies
    if dmode == 'RNASeq':
        ANNOT = pd.read_table(f'{IN}/{dmode}/GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt', index_col=0)
        SMTSD = ANNOT.loc[data.columns.get_level_values(0),'SMTSD']
        MI = pd.MultiIndex.from_tuples(zip(data.columns,SMTSD))    
        data.columns = MI
    else:
        ANNOT = pd.read_csv(f'{IN}/{dmode}/GSM_to_CellType_final.csv', index_col=0)
        VCS = ANNOT.CT.value_counts()
        CT = VCS[VCS>37].index.tolist()
        gsm_ct_l = []
        for gsm,ct in ANNOT.CT.items():
            ctp = ct if ct in CT else 'Other'
            gsm_ct_l.append((gsm,ctp))
        MI = pd.MultiIndex.from_tuples(gsm_ct_l)
        data.columns = MI
    return data

if dmode=='RNASeq':
    proj_data = pd.read_csv(f'{IN}/{dmode}/{DATA_FN}',header=0,dtype=str).set_index('Unnamed: 0').astype(float).T
    proj_deltas = pd.read_csv(f'{IN}/{dmode}/{DELTA_FN}',header=[0,1,2],dtype=str).set_index([('Unnamed: 0_level_0','Unnamed: 0_level_1','Unnamed: 0_level_2')]).astype(float)
else:
    proj_data = pd.read_csv(f'{IN}/{dmode}/{DATA_FN}',dtype=str).set_index(['Unnamed: 0','Unnamed: 1']).astype(float)
    proj_deltas = pd.read_csv(f'{IN}/{dmode}/{DELTA_FN}',dtype=str).set_index('Unnamed: 0').astype(float)
    proj_deltas = proj_deltas.loc[feat_l[:4]]
    


proj_deltas = proj_deltas.sort_index(axis=1)
proj_data = fix_indices(proj_data.T).T
cts = proj_data.index.get_level_values(1).unique()
cts = cts[cts!='None']
CAT = pd.Categorical(proj_data.index.get_level_values(level=1))
if not osp.exists(f'{IN}/{dmode}/KNNGP2NM_ser.csv'):
    pd.Series(dict(zip(CAT.codes,CAT.tolist()))).to_csv(f'{IN}/{dmode}/KNNGP2NM_ser.csv')
idx = CAT.categories
Y = CAT.codes
WT = weights(proj_data)    
scaled_states =proj_data.multiply(WT)
clf = KNC(n_neighbors, weights='distance')
clf.fit(scaled_states.values,Y)

def main():
    FS()

if __name__ == '__main__':
    main()
