from ase.db import connect
from icet import (StructureContainer,
                  CrossValidationEstimator,
                  ClusterExpansion)
from mchammer.ensembles import SemiGrandCanonicalEnsemble as SGCEnsemble
from mchammer.calculators import ClusterExpansionCalculator as CECalculator
from mchammer.observers import ClusterExpansionObserver as CEObserver

# step 1: Read cluster expansion for mixing energies from file
ce_mix_energies = ClusterExpansion.read('mixing_energy.ce')
chemical_symbols = ce_mix_energies.cluster_space.chemical_symbols[0]
cs = ce_mix_energies.cluster_space

# step 2: Read structures from database into a structure container
db = connect('reference_data.db')
sc = StructureContainer(cluster_space=cs)
for row in db.select('natoms<=8'):
    sc.add_structure(structure=row.toatoms(),
                     user_tag=row.tag,
                     properties={'lattice_parameter': row.lattice_parameter})

# step 3: Construct cluster expansion for lattice parameter
fit_data = sc.get_fit_data(key='lattice_parameter')
opt = CrossValidationEstimator(fit_data=fit_data, fit_method='lasso')
opt.validate()
opt.train()
ce_latt_param = ClusterExpansion(cluster_space=cs, parameters=opt.parameters)

# step 4: Set up the calculator and a canonical ensemble
structure = cs.primitive_structure.repeat(3)
structure.set_chemical_symbols([chemical_symbols[0]] * len(structure))
calculator = CECalculator(structure=structure, cluster_expansion=ce_mix_energies)
ensemble = SGCEnsemble(calculator=calculator, structure=structure,
                       random_seed=42, temperature=900.0,
                       chemical_potentials={'Ag': 0, 'Pd': 0},
                       ensemble_data_write_interval=10)

# step 5: Attach observer and run
observer = CEObserver(cluster_expansion=ce_latt_param, interval=10)
ensemble.attach_observer(observer=observer, tag='lattice_parameter')
ensemble.run(number_of_trial_steps=1000)

# step 6: Print data
print(ensemble.data_container.data)
