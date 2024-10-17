# SMS_builder. Build SMS++ and test its functionalities

[![Status Linux](https://github.com/SPSUnipi/SMSpp_builder/actions/workflows/build-linux.yml/badge.svg)](https://github.com/SPSUnipi/SMSpp_builder/actions/workflows/build-linux.yml)

This repo builds [SMS++](https://gitlab.com/smspp/smspp-project) and tests its functionalities twice a week.
The procedure that is followed is:
- Install the main dependencies
- Clone the SMS++ repository, develop branch
- Build SMS++
- Test the installation

The intallation procedure adopted in the CI is available on the [Installing SMS++ web page](https://gitlab.com/smspp/smspp-project/-/wikis/Installing-SMS++), including SCIP, CPLEX, Gurobi and HiGHS solvers.

Moreover, the procedure tests the successful execution of:
- the Unit Commitment Solver [UCBlockSolver](https://gitlab.com/smspp/tools/-/tree/develop/ucblock_solver)
- the Investment Block Solver, currently using the test in [InvestmentBlock test](https://gitlab.com/smspp/investmentblock/-/tree/master/test)