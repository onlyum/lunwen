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
        Name = "E3_pgfm_center0p2_edge0p1_bce"
        Args = @(
            "--experiment_name", "E3_pgfm_center0p2_edge0p1_bce",
            "--loss_mode", "bce",
            "--use_prior",
            "--use_aux",
            "--center_weight", "0.2",
            "--edge_weight", "0.1",
            "--prior_gamma_init", "0.1",
            "--fa_weight", "0.0",
            "--center_mode", "avg"
        )
    },
    @{
        Name = "E6_pgfm_center0p1_edge0p05_bce"
        Args = @(
            "--experiment_name", "E6_pgfm_center0p1_edge0p05_bce",
            "--loss_mode", "bce",
            "--use_prior",
            "--use_aux",
            "--center_weight", "0.1",
            "--edge_weight", "0.05",
            "--prior_gamma_init", "0.1",
            "--fa_weight", "0.0",
            "--center_mode", "avg"
        )
    },
    @{
        Name = "E7_pgfm_gamma0p05_aux_bce"
        Args = @(
            "--experiment_name", "E7_pgfm_gamma0p05_aux_bce",
            "--loss_mode", "bce",
            "--use_prior",
            "--use_aux",
            "--center_weight", "0.2",
            "--edge_weight", "0.1",
            "--prior_gamma_init", "0.05",
            "--fa_weight", "0.0",
            "--center_mode", "avg"
        )
    },
    @{
        Name = "E12_pgfm_gaussian_center_edge_bce"
        Args = @(
            "--experiment_name", "E12_pgfm_gaussian_center_edge_bce",
            "--loss_mode", "bce",
            "--use_prior",
            "--use_aux",
            "--center_weight", "0.2",
            "--edge_weight", "0.1",
            "--prior_gamma_init", "0.1",
            "--fa_weight", "0.0",
            "--center_mode", "gaussian"
        )
    },
    @{
        Name = "E13_pgfm_aux_valley_loss"
        Args = @(
            "--experiment_name", "E13_pgfm_aux_valley_loss",
            "--loss_mode", "bce",
            "--use_prior",
            "--use_aux",
            "--center_weight", "0.2",
            "--edge_weight", "0.1",
            "--prior_gamma_init", "0.1",
            "--fa_weight", "0.0",
            "--center_mode", "avg",
            "--valley_weight", "0.002"
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

Write-Output "===== PRIORITY_SEQUENCE_DONE $(Get-Date -Format "yyyy-MM-dd HH:mm:ss") ====="
