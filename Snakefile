from pathlib import Path

if "config" not in globals() or not config:
    fp = "configs/T_1N.yaml"
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
        "resources/networks/microgrid" + SNAME + ".nc"
    log:
        "logs/builder" + SNAME + ".log"
    script:
        "scripts/microgrid_builder.py"

rule mirogrid_optimizer:
    params:
        solver_name = config['solver_name'],
        solver_options = config.get('solver_options', {}),
    input:
        "resources/networks/microgrid" + SNAME + ".nc"
    output:
        "results/networks/microgrid" + SNAME + "_optimized.nc"
    log:
        "logs/optimizer" + SNAME + ".log"
    script:
        "scripts/microgrid_optimizer.py"

rule smspp_dispatch_builder:
    params:
        block_config=config['block_config'],
    input:
        "results/networks/microgrid" + SNAME + "_optimized.nc"
    output:
        "resources/smspp/microgrid" + SNAME + ".nc4"
    log:
        "logs/smspp_dispatch_builder" + SNAME + ".log"
    script:
        "scripts/smspp_dispatch_builder.py"

rule smspp_investment_builder:
    params:
        block_config=config['block_config'],
    input:
        "results/networks/microgrid" + SNAME + "_optimized.nc"
    output:
        "resources/smspp/microgrid" + SNAME + "_investment.nc4"
    log:
        "logs/smspp_investment_builder" + SNAME + ".log"
    script:
        "scripts/smspp_investment_builder.py"

rule smspp_dispatch_optimizer:
    input:
        smspp_file="resources/smspp/microgrid" + SNAME + ".nc4",
        configdir="data/SMSpp/UCBlockSolver/",
        config="data/SMSpp/UCBlockSolver/uc_solverconfig.txt",
    output:
        "results/smspp/microgrid" + SNAME + "_optimized.txt"
    log:
        "logs/smspp_dispatch_optimizer" + SNAME + ".log"
    shell:
        "ucblock_solver {input.smspp_file} -c {input.configdir} -S {input.config} >> {output}"

rule smspp_investment_optimizer:
    input:
        smspp_file="resources/smspp/microgrid" + SNAME + "_investment.nc4",
        configdir="data/SMSpp/InvestmentBlockTest/config",
        config="data/SMSpp/InvestmentBlockTest/config/BSPar.txt",
    output:
        "results/smspp/microgrid" + SNAME + "_investment_optimized.txt"
    log:
        "logs/smspp_dispatch_optimizer" + SNAME + ".log"
    shell:
        "InvestmentBlock_test {input.smspp_file} -c {input.configdir}/ -S {input.config} >> {output}"

rule verify_dispatch:
    params:
        tolerances=config['tolerances']
    input:
        pypsa_network="results/networks/microgrid" + SNAME + "_optimized.nc",
        smspp_log="results/smspp/microgrid" + SNAME + "_optimized.txt",
    output:
        touch("results/microgrid" + SNAME + "_complete.txt")
    log:
        "logs/verify_dispatch" + SNAME + ".log"
    script:
        "scripts/verify_dispatch.py"

rule verify_investment:
    params:
        tolerances=config['tolerances']
    input:
        pypsa_network="results/networks/microgrid" + SNAME + "_optimized.nc",
        smspp_log="results/smspp/microgrid" + SNAME + "_investment_optimized.txt",
    output:
        touch("results/microgrid" + SNAME + "_investment_complete.txt")
    log:
        "logs/verify_investment" + SNAME + "_investment.log"
    script:
        "scripts/verify_investment.py"

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

rule run_all_investment:
    run:
        import subprocess
        import sys
        for s_name in Path("configs").glob("*.yaml"):
            print(str(s_name))
            subprocess.run(
                f"snakemake -j1 verify_investment --configfile {str(s_name)} --force",
                shell=True,
                check=True,
                stdout=sys.stdout,
            )

rule dag:
    output:
        dot="results/plots/dag/dag.dot",
        pdf="results/plots/dag/dag.pdf",
        png="results/plots/dag/dag.png",
    shell:
        r"""
        snakemake --rulegraph verify_dispatch | sed -n "/digraph/,\$p" > {output.dot}
        dot -Tpdf -o {output.pdf} {output.dot}
        dot -Tpng -o {output.png} {output.dot}
        """
