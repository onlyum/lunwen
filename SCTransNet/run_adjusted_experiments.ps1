$ErrorActionPreference = "Stop"

$Python = "D:\ProgramFiles\IT\Envs\Anaconda\envs\lunwen\python.exe"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$Common = @(
    "--batchSize", "16",
    "--epochs", "1000",
    "--begin_val", "1",
    "--val_interval", "1",
    "--test_interval", "5",
    "--every_print", "1",
    "--early_stop_patience", "10",
    "--min_epochs", "15",
    "--train_loader_variant", "normal"
)

$Experiments = @(
    @{
        Name = "E12b_pgfm_adaptive_gaussian_center_edge_bce"
        Args = @(
            "--experiment_name", "E12b_pgfm_adaptive_gaussian_center_edge_bce",
            "--loss_mode", "bce",
            "--use_prior",
            "--use_aux",
            "--center_weight", "0.2",
            "--edge_weight", "0.1",
            "--prior_gamma_init", "0.1",
            "--fa_weight", "0.0",
            "--center_mode", "gaussian",
            "--center_dirac_area", "2",
            "--center_min_sigma", "0.5",
            "--center_max_sigma", "1.5"
        )
    },
    @{
        Name = "E13b_pgfm_aux_dt_repulsion"
        Args = @(
            "--experiment_name", "E13b_pgfm_aux_dt_repulsion",
            "--loss_mode", "bce",
            "--use_prior",
            "--use_aux",
            "--center_weight", "0.2",
            "--edge_weight", "0.1",
            "--prior_gamma_init", "0.1",
            "--fa_weight", "0.0",
            "--center_mode", "avg",
            "--valley_weight", "0.002",
            "--valley_max_distance", "24.0",
            "--repulsion_sigma", "5.0",
            "--repulsion_w0", "1.0"
        )
    }
)

foreach ($Experiment in $Experiments) {
    $StartTime = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Output "===== EXPERIMENT_START $($Experiment.Name) $StartTime ====="
    $ArgsList = @("-u", "train.py") + $Common + $Experiment.Args
    & $Python @ArgsList
    $ExitCode = $LASTEXITCODE
    $EndTime = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Output "===== EXPERIMENT_END $($Experiment.Name) exit=$ExitCode $EndTime ====="
    if ($ExitCode -ne 0) {
        exit $ExitCode
    }
}

Write-Output "===== ADJUSTED_SEQUENCE_DONE $(Get-Date -Format "yyyy-MM-dd HH:mm:ss") ====="
