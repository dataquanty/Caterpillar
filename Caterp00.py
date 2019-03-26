# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 15:39:01 2015

@author: dataquanty
"""

import numpy as np
import pandas as pd
import glob
from vectorizer import vectorize, kpca_vector
from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import scale


mat1 = pd.read_csv('competition_data/train_set.csv')
mat2 = pd.read_csv('competition_data/test_set.csv')
mat = pd.concat([mat1,mat2])
mat['bracket_pricing']=mat['bracket_pricing'].apply(lambda row: 1 if row=='Yes' else 0)
mat['quantity']=mat['quantity']*mat['bracket_pricing']+mat['min_order_quantity']*(1-mat['bracket_pricing'])
mat.drop('min_order_quantity',inplace=True,axis=1)


tube = pd.read_csv('competition_data/tube.csv')

mat = mat.merge(tube,how='left',left_on='tube_assembly_id',right_on='tube_assembly_id')

mat['volume']=((mat['diameter']/2.0)**2*np.pi)*mat['length']
mat['plain_volume']=mat['volume']-np.pi*((mat['diameter']/2.0-mat['wall'])**2)*mat['length']

cols = ['end_a_1x','end_a_2x','end_x_1x','end_x_2x']
for c in cols:
    mat[c]=mat[c].apply(lambda row : 1 if row == 'Y' else 0)


# BOM

lstfiles = glob.glob('competition_data/comp/*')
comp = pd.DataFrame()

for f in lstfiles:
    comp = pd.concat([comp,pd.read_csv(f)])

bom = pd.read_csv('competition_data/bill_of_materials.csv')

bomaggf = pd.DataFrame()
for i in range(1,9):
    bom.set_index(['tube_assembly_id','component_id_'+str(i)],inplace=True)
    bomagg = bom['quantity_'+str(i)].reset_index()
    bomagg.columns = ['tube_assembly_id','component_id','quantity']
    bomagg = bomagg[~np.isnan(bomagg['quantity'])]
    bomaggf = pd.concat([bomaggf, bomagg])
    bom.reset_index(inplace=True)


comp['groove']=comp['groove'].apply(lambda row: 1 if row=='Yes' else 0)
comp['orientation']=comp['orientation'].apply(lambda row: 1 if row=='Yes' else 0)
comp['unique_feature']=comp['unique_feature'].apply(lambda row: 1 if row=='Yes' else 0)

def uniq(x):
    return np.size(np.unique(x))
   
compagg = {'component_id': uniq,
           'weight':np.sum,
           'component_type_id':uniq,
           'bolt_pattern_long':np.sum,
           'bolt_pattern_wide':np.sum,
           'drop_length':np.sum,
           'elbow_angle':np.sum,
           'extension_length':np.sum,
           'groove':np.sum,
           'orientation':np.max,
           'overall_length':np.sum,
           'thickness':np.sum,
           'unique_feature':np.sum,
           }

bomaggf = bomaggf.merge(comp[compagg.keys()],how='left',left_on='component_id',right_on='component_id')
bomaggf['weight_tot']=bomaggf['quantity']*bomaggf['weight']

#missing weights !!!! 

aggfunc = {'quantity':np.sum, 
           'weight_tot':np.sum}
aggfunc.update(compagg)

"""
aggfunc = {'component_id':uniq,
           'quantity':np.sum, 
           'weight_tot':np.sum,
           'component_type_id':uniq}
"""
        
bomaggfgr = bomaggf.groupby('tube_assembly_id').agg(aggfunc)
bomaggfgr.reset_index(inplace=True)

bomaggfgr.columns = [c.replace('quantity','n_components') for c in bomaggfgr.columns]
mat = mat.merge(bomaggfgr, how='left',left_on='tube_assembly_id',right_on='tube_assembly_id')

"""
cols = ['component_id']
vectorSpec = vectorize(bomaggf,cols,0.003)
bomaggf = pd.concat([bomaggf,vectorSpec],axis=1)
bomaggf.drop(cols+['quantity'],axis=1,inplace=True)
bomaggf = bomaggf.groupby('tube_assembly_id').sum().reset_index()

mat = mat.merge(bomaggf,how='left',left_on='tube_assembly_id',right_on='tube_assembly_id')
"""


# SPECS
specs = pd.read_csv('competition_data/specs.csv')

specs = specs.set_index('tube_assembly_id')
specs = specs.stack()
specs = specs.reset_index()
specs.drop('level_1',inplace=True,axis=1)
specs.columns = ['tube_assembly_id','specs']

nspecs = specs.groupby('tube_assembly_id').count().reset_index()




cols = ['specs']
vectorSpec = vectorize(specs,cols,0.001)
specs = pd.concat([specs,vectorSpec],axis=1)
specs.drop(cols,axis=1,inplace=True)

specs = specs.groupby('tube_assembly_id').sum()

n_comp=3
kern = scale(np.array(specs,dtype=np.float))
kpca = RBFSampler(n_components=n_comp, gamma=0.3)
kern = kpca.fit_transform(kern)
specs.drop(specs.columns,axis=1,inplace=True)
cols = ['specs_pca'+ str(i) for i in range(n_comp)]
for c in cols:
    specs[c]=np.zeros(len(specs))

specs[cols]=kern
specs.reset_index(inplace=True)

mat = mat.merge(specs,how='left',left_on='tube_assembly_id',right_on='tube_assembly_id')
mat = mat.merge(nspecs,how='left',left_on='tube_assembly_id',right_on='tube_assembly_id')

material = kpca_vector(mat,['material_id'],0.3)
material.columns = ['material_'+ c for c in material.columns]
mat.drop('material_id',inplace=True,axis=1)
mat = pd.concat([mat,material],axis=1)


mat.to_csv('data.csv',index=False)

#study cost
"""
#mat['cost_adjusted']=mat['cost']/mat['quantity']
mat['base_cost']=mat['cost']*mat['quantity'].apply(lambda row: 1 if row==1 else np.nan)
mat['base_cost']=mat['base_cost'].fillna(method='ffill')
mat['cost_index']=100*mat['cost']/mat['base_cost']

#mat['quantity']=mat['minç]
matcost = mat[['tube_assembly_id','quantity','cost_index']][np.isnan(mat['id'])].copy()
matcost = matcost.groupby(['tube_assembly_id','quantity']).mean()
matcost = matcost.reset_index()
matcost.drop_duplicates(inplace=True)
matcost = matcost.set_index(['tube_assembly_id','quantity'])
"""
