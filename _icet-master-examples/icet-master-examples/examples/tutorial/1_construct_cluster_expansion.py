# This scripts runs in about 16 seconds on an i7-6700K CPU.

from ase.db import connect
from icet import ClusterSpace, StructureContainer, ClusterExpansion
from trainstation import CrossValidationEstimator

# step 1: Basic setup
db = connect('reference_data.db')
primitive_structure = db.get(id=1).toatoms()  # primitive structure

# step 2: Set up the basic structure and a cluster space
cs = ClusterSpace(structure=primitive_structure,
                  cutoffs=[13.5, 6.0, 5.5],
                  chemical_symbols=['Ag', 'Pd'])
print(cs)

# step 3: Parse the input structures and set up a structure container
sc = StructureContainer(cluster_space=cs)
for row in db.select('natoms<=8'):
    sc.add_structure(structure=row.toatoms(),
                     user_tag=row.tag,
                     properties={'mixing_energy': row.mixing_energy})
print(sc)

# step 4: Train parameters
opt = CrossValidationEstimator(fit_data=sc.get_fit_data(key='mixing_energy'),
                               fit_method='lasso')
opt.validate()
opt.train()
print(opt)

# step 5: Construct cluster expansion and write it to file
ce = ClusterExpansion(cluster_space=cs, parameters=opt.parameters, metadata=opt.summary)
print(ce)
ce.write('mixing_energy.ce')
