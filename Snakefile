from pathlib import Path

if "config" not in globals() or not config:
    fp = "configs/microgrid_T_1N.yaml"
    print("No config file specified. Using default config file: " + fp + "\n")
    configfile: fp

SNAME = "" if config.get('name', "") == "" else "_" + config['name']

rule microgrid_builder:
    params:
        model_parameters = config['model_parameters'],
        n_snapshots = config['n_snapshots'],
        apply_weighting_patch = config.get('apply_weighting_patch', False),
    input:
        pv_file = 'data/microgrid/ninja_pv_1.7250_33.6208_uncorrected.csv',
        wind_file = 'data/microgrid/ninja_wind_1.7250_33.6208_uncorrected.csv',
        demand_file = 'data/microgrid/demand_data.csv',
    output:
        "resources/networks/microgrid_" + SNAME + ".nc"
    log:
        "logs/microgrid_builder_" + SNAME + ".log"
    script:
        "scripts/microgrid_builder.py"

rule microgrid_optimizer:
    params:
        solver_name = config['solver_name'],
        solver_options = config.get('solver_options', {}),
    input:
        "resources/networks/microgrid_" + SNAME + ".nc"
    output:
        "results/networks/microgrid_" + SNAME + "_optimized.nc"
    log:
        "logs/microgrid_optimizer_" + SNAME + ".log"
    script:
        "scripts/microgrid_optimizer.py"

rule smspp_dispatch_builder:
    input:
        "results/networks/microgrid_" + SNAME + "_optimized.nc"
    output:
        "resources/smspp/microgrid_" + SNAME + ".nc4"
    log:
        "logs/smspp_dispatch_builder_" + SNAME + ".log"
    script:
        "scripts/smspp_dispatch_builder.py"

rule smspp_dispatch_optimizer:
    input:
        smspp_file="resources/smspp/microgrid_" + SNAME + ".nc4",
        configdir="data/SMSpp/UCBlockSolver/",
        config="data/SMSpp/UCBlockSolver/uc_solverconfig.txt",
    output:
        "results/smspp/microgrid_" + SNAME + "_optimized.txt"
    log:
        "logs/smspp_dispatch_optimizer_" + SNAME + ".log"
    shell:
        "ucblock_solver {input.smspp_file} -c {input.configdir} -S {input.config} >> {output}"

rule verify_dispatch:
    params:
        tolerances=config['tolerances']
    input:
        pypsa_network="results/networks/microgrid_" + SNAME + "_optimized.nc",
        smspp_log="results/smspp/microgrid_" + SNAME + "_optimized.txt",
    output:
        touch("results/microgrid_" + SNAME + "_complete.txt")
    log:
        "logs/verify_dispatch_" + SNAME + ".log"
    script:
        "scripts/verify_dispatch.py"

rule run_all_dispatch:
    run:
        import subprocess
        import sys
        for s_name in Path("configs").glob("*.yaml"):
            print(str(s_name))
            subprocess.run(
                f"snakemake -j1 verify_dispatch --configfile {str(s_name)} --force",
                shell=True,
                check=True,
                stdout=sys.stdout,
            )