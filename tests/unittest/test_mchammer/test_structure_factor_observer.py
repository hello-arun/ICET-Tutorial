import numpy as np
import pytest
from ase.build import bulk
from contextlib import contextmanager
from mchammer.observers import StructureFactorObserver
from mchammer.observers.base_observer import BaseObserver
from tempfile import NamedTemporaryFile
from icet.input_output.logging_tools import logger, set_log_config


# test the various ways the initialization can fail
@contextmanager
def does_not_raise():
    yield


@pytest.fixture
def system(request):
    model, repeat = request.param
    if model == 'AlCu_fcc':
        alat = 3.1
        blat = clat = alat
        prim = bulk('Al', a=alat, crystalstructure='fcc')
        supercell = prim.repeat(repeat)
        ns = int(0.3 * len(supercell))
        assert ns > 0
        supercell.symbols[:ns] = 'Cu'
    elif model == 'AuCu3_L12':
        alat = 3.2
        blat = clat = alat
        prim = bulk('Cu', a=alat, crystalstructure='fcc', cubic=True)
        prim[0].symbol = 'Au'
        supercell = prim.repeat(repeat)
    elif model == 'AlCuMg_fcc':
        alat = 3.4
        blat = clat = alat
        prim = bulk('Al', a=alat, crystalstructure='fcc', cubic=True)
        prim[0].symbol = 'Cu'
        prim[1].symbol = 'Mg'
        supercell = prim.repeat(repeat)
    elif model == 'AgPd_hcp':
        alat, clat = 3.4, 5.1
        blat = alat
        prim = bulk('Ag', a=alat, c=clat, crystalstructure='hcp')
        prim[0].symbol = 'Pd'
        supercell = prim.repeat(repeat)
    elif model == 'Al_fcc':
        alat = 3.1
        blat = clat = alat
        prim = bulk('Al', a=alat, crystalstructure='fcc')
        supercell = prim.repeat(repeat)
    else:
        raise ValueError(f'Unknown model: {model}')

    pairs = [(e1, e2) for e1 in set(supercell.symbols) for e2 in set(supercell.symbols)]

    q_points = []
    q_points.append(2 * np.pi / alat * np.array([1, 0, 0]))
    q_points.append(2 * np.pi / blat * np.array([0, 1, 0]))
    q_points.append(2 * np.pi / clat * np.array([0, 0, 1]))
    q_points.append(2 * np.pi / alat * np.array([1 / 2, 0, 0]))
    q_points.append(2 * np.pi / alat * np.array([1 / 3, 1 / 3, 0]))

    return prim, supercell, q_points, pairs


@pytest.mark.parametrize('system', [
        (('AlCu_fcc', 2)),
        (('AuCu3_L12', 2)),
        (('AlCuMg_fcc', 2)),
        (('AgPd_hcp', 2)),
    ], indirect=['system'])
def test_initialization_basic(system):
    _, supercell, q_points, pairs = system
    sfo = StructureFactorObserver(supercell, q_points, interval=789)
    assert isinstance(sfo, BaseObserver)
    assert sfo.interval == 789
    assert all([s in pairs for s in sfo._pairs])
    assert all([s in sfo.form_factors and sfo.form_factors[s] == 1
                for s in set(supercell.symbols)])
    assert np.all(sfo.q_points == q_points)
    assert 'q-point' in str(sfo)
    assert 'StructureFactorObserver' in str(sfo)


@pytest.mark.parametrize('system,pairs,expectation', [
        (('AlCu_fcc', 2), [('Al', 'Al'), ('Al', 'Cu'), ('Cu', 'Cu')], does_not_raise()),
        (('AlCu_fcc', 2), [('Al', 'Al'), ('Cu', 'Al'), ('Cu', 'Cu')], does_not_raise()),
        (('AlCu_fcc', 2), [('Al', 'Al'), ('Al', 'Cu'), ('Cu', 'Al'), ('Cu', 'Cu')],
         does_not_raise()),
        (('AlCu_fcc', 2), [('Al', 'Si')], does_not_raise()),
        (('AlCu_fcc', 2), [('Al', 'Z')], pytest.raises(ValueError)),
        (('AlCu_fcc', 2), [], pytest.raises(ValueError)),
    ], indirect=['system'])
def test_initialization_pairs(system, pairs, expectation):
    _, supercell, q_points, _ = system
    with expectation:
        _ = StructureFactorObserver(supercell, q_points, symbol_pairs=pairs)


@pytest.mark.parametrize('system, pairs, n_warnings, warning', [
    (('Al_fcc', 2), None, 1, "Only one pair requested ('Al', 'Al')"),
    (('AlCu_fcc', 2), [('Al', 'Al')], 1, "Only one pair requested ('Al', 'Al')"),
    (('AlCu_fcc', 2), [('Al', 'Cu')], 0, None),
], indirect=['system'])
def test_initialization_warning(system, pairs, n_warnings, warning):
    _, supercell, q_points, _ = system

    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)

    logfile = NamedTemporaryFile(mode='w+', encoding='utf-8', delete=False)
    set_log_config(filename=logfile.name)
    StructureFactorObserver(supercell, q_points, symbol_pairs=pairs)

    logfile.seek(0)
    lines = logfile.readlines()
    logfile.close()
    assert len(lines) == n_warnings
    if n_warnings > 0:
        assert warning in lines[0]


@pytest.mark.parametrize('system,form_factors,expectation', [
        (('AlCu_fcc', 2), dict(Al=1, Cu=2), does_not_raise()),
        (('AlCu_fcc', 2), dict(Al=1), pytest.raises(ValueError)),
        (('AlCu_fcc', 2), dict(Al=1, Cu=0), pytest.raises(ValueError)),
    ], indirect=['system'])
def test_initialization_form_factors(system, form_factors, expectation):
    _, supercell, q_points, _ = system
    with expectation:
        _ = StructureFactorObserver(supercell, q_points, form_factors=form_factors)


@pytest.mark.parametrize('system', [
        (('AlCu_fcc', 2)),
        (('AuCu3_L12', 2)),
        (('AlCuMg_fcc', 2)),
        (('AgPd_hcp', 2)),
    ], indirect=['system'])
def test_get_observable(system):
    prim, supercell, q_points, _ = system
    sfo = StructureFactorObserver(supercell, q_points)
    with does_not_raise():
        sfo.get_observable(supercell)
    with pytest.raises(ValueError):
        sfo.get_observable(prim)
    with does_not_raise():
        supercell[0].symbol = 'X'
        sfo.get_observable(supercell)
    with does_not_raise():
        supercell.symbols = prim[0].symbol
        sfo.get_observable(supercell)
    with does_not_raise():
        supercell.symbols = 'W'
        sfo.get_observable(supercell)


expected_results = [
    (('AlCu_fcc', 2), dict(sfo_Al_Al_q0=0.0,
                           sfo_Al_Al_q1=0.0,
                           sfo_Al_Al_q2=0.5,
                           sfo_Al_Al_q3=2.25,
                           sfo_Al_Al_q4=2.375,
                           sfo_Al_Cu_q0=0.0,
                           sfo_Al_Cu_q1=0.0,
                           sfo_Al_Cu_q2=-1.0,
                           sfo_Al_Cu_q3=1.5,
                           sfo_Al_Cu_q4=2.0,
                           sfo_Cu_Cu_q0=0.0,
                           sfo_Cu_Cu_q1=0.0,
                           sfo_Cu_Cu_q2=0.5,
                           sfo_Cu_Cu_q3=0.25,
                           sfo_Cu_Cu_q4=0.125)),
    (('AuCu3_L12', 2), dict(sfo_Au_Au_q0=2.0,
                            sfo_Au_Au_q1=2.0,
                            sfo_Au_Au_q2=2.0,
                            sfo_Au_Au_q3=0.0,
                            sfo_Au_Au_q4=-0.25,
                            sfo_Au_Cu_q0=-4.0,
                            sfo_Au_Cu_q1=-4.0,
                            sfo_Au_Cu_q2=-4.0,
                            sfo_Au_Cu_q3=0.0,
                            sfo_Au_Cu_q4=0.5,
                            sfo_Cu_Cu_q0=2.0,
                            sfo_Cu_Cu_q1=2.0,
                            sfo_Cu_Cu_q2=2.0,
                            sfo_Cu_Cu_q3=0.0,
                            sfo_Cu_Cu_q4=1.71875)),
    (('AlCuMg_fcc', 2), dict(sfo_Al_Al_q0=8.0,
                             sfo_Al_Al_q1=0.0,
                             sfo_Al_Al_q2=0.0,
                             sfo_Al_Al_q3=0.0,
                             sfo_Al_Al_q4=0.0,
                             sfo_Al_Cu_q0=-8.0,
                             sfo_Al_Cu_q1=0.0,
                             sfo_Al_Cu_q2=0.0,
                             sfo_Al_Cu_q3=-0.0,
                             sfo_Al_Cu_q4=0.75,
                             sfo_Al_Mg_q0=-8.0,
                             sfo_Al_Mg_q1=0.0,
                             sfo_Al_Mg_q2=0.0,
                             sfo_Al_Mg_q3=-0.0,
                             sfo_Al_Mg_q4=1.3125,
                             sfo_Cu_Cu_q0=2.0,
                             sfo_Cu_Cu_q1=2.0,
                             sfo_Cu_Cu_q2=2.0,
                             sfo_Cu_Cu_q3=0.0,
                             sfo_Cu_Cu_q4=-0.25,
                             sfo_Cu_Mg_q0=4.0,
                             sfo_Cu_Mg_q1=-4.0,
                             sfo_Cu_Mg_q2=-4.0,
                             sfo_Cu_Mg_q3=0.0,
                             sfo_Cu_Mg_q4=-0.25,
                             sfo_Mg_Mg_q0=2.0,
                             sfo_Mg_Mg_q1=2.0,
                             sfo_Mg_Mg_q2=2.0,
                             sfo_Mg_Mg_q3=0.0,
                             sfo_Mg_Mg_q4=-0.25)),
    (('AgPd_hcp', 2), dict(sfo_Ag_Ag_q0=0.0,
                           sfo_Ag_Ag_q1=3.332262,
                           sfo_Ag_Ag_q2=4.0,
                           sfo_Ag_Ag_q3=0.0,
                           sfo_Ag_Ag_q4=0.259381,
                           sfo_Ag_Pd_q0=0.0,
                           sfo_Ag_Pd_q1=-3.775374,
                           sfo_Ag_Pd_q2=-8.0,
                           sfo_Ag_Pd_q3=1.0,
                           sfo_Ag_Pd_q4=0.497788,
                           sfo_Pd_Pd_q0=0.0,
                           sfo_Pd_Pd_q1=3.332262,
                           sfo_Pd_Pd_q2=4.0,
                           sfo_Pd_Pd_q3=0.0,
                           sfo_Pd_Pd_q4=0.259381)),
    ]


@pytest.mark.parametrize('system,expected_outcome', [
        (expected_results[0][0], expected_results[0][1]),
        (expected_results[1][0], expected_results[1][1]),
        (expected_results[2][0], expected_results[2][1]),
        (expected_results[3][0], expected_results[3][1]),
    ], indirect=['system'])
def test_sfo_values(system, expected_outcome):
    _, supercell, q_points, _ = system
    sfo = StructureFactorObserver(supercell, q_points)
    res = sfo.get_observable(supercell)
    for key, target_value in expected_outcome.items():
        assert np.isclose(target_value, res[key], atol=1e-6)
