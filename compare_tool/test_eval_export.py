"""Unit tests for the eval-export helpers (pure, no Flask)."""

from eval_export import (collect_eval_rows, build_csv, build_sh,
                         command_for_winner)

# A finetune winner GIF: target T001 finetuned from T000 starting at epoch 140.
FT = ('ls3639/NM0029/finetune/'
      'NM0029_HT_100nm_T001_0001_rec__from_NM0029_HT_100nm_T000_0001_rec__140/'
      'all_epochs_view2_141-150.gif')
# A second finetune of the same target from a different checkpoint (T014@170).
FT2 = ('ls3639/NM0029/finetune/'
       'NM0029_HT_100nm_T001_0001_rec__from_NM0029_HT_100nm_T014_0001_rec__170/'
       'all_epochs_view2_171-180.gif')
# A plain training tile (no `_from_` checkpoint).
TR = 'ls3639/NM0029/NM0029_HT_100nm_T000_0001_rec_/all_epochs_view0_0-45.gif'


def _result(winner_gif, ep, gifs=None, extra=''):
    top3 = f'{winner_gif} Ep.{ep}'
    if extra:
        top3 += ', ' + extra
    return {'gifs': gifs or [winner_gif], 'top3': top3}


def test_finetune_winner_fields():
    (r,) = collect_eval_rows([_result(FT, 142)])
    assert r.tile == 'T001'
    assert r.full_metadata == 'NM0029_HT_100nm_T001_0001_rec_'
    assert r.finetune_tile == 'T000'
    assert r.selected_epoch == 142
    assert r.full_checkpoint_metadata == 'NM0029_HT_100nm_T000_0001_rec_'
    assert r.finetune_epoch == 140


def test_two_digit_tile_no_subid():
    gif = ('ls3543/finetune/S1M_RH_rot_HTb_100nm_T01_rec__from_'
           'S1M_RH_rot_HTb_100nm_T35_rec__195/all_epochs_view0_196-205.gif')
    (r,) = collect_eval_rows([_result(gif, 198)])
    assert r.tile == 'T01'
    assert r.full_metadata == 'S1M_RH_rot_HTb_100nm_T01_rec_'
    assert r.finetune_tile == 'T35'
    assert r.full_checkpoint_metadata == 'S1M_RH_rot_HTb_100nm_T35_rec_'
    assert r.selected_epoch == 198
    assert r.finetune_epoch == 195


def test_tile_token_with_resolution_segment():
    # Tile not directly before _rec_: `T03_050nm_rec_` (A033 naming).
    gif = ('A033/finetune/A033_T03_050nm_rec__from_A033_T02_050nm_rec__170/'
           'all_epochs_view0_171-180.gif')
    (r,) = collect_eval_rows([_result(gif, 180)])
    assert r.tile == 'T03'
    assert r.finetune_tile == 'T02'
    assert r.full_metadata == 'A033_T03_050nm_rec_'
    assert r.finetune_epoch == 170


def test_tile_token_ignores_HT_prefix():
    # `HT` must not be read as a tile; the real tile is T002.
    gif = ('NM0024/finetune/NM0024_HT_090nm_T002_0001_rec__from_'
           'NM0024_HT_090nm_T004_0001_rec__115/all_epochs_view1_116-125.gif')
    (r,) = collect_eval_rows([_result(gif, 123)])
    assert r.tile == 'T002'
    assert r.finetune_tile == 'T004'


def test_training_tile_skipped():
    assert collect_eval_rows([_result(TR, 30)]) == []


def test_multi_gif_winner_drives_row():
    # Winner is FT2 even though FT is listed first in gifs.
    res = {'gifs': [FT, FT2], 'top3': f'{FT2} Ep.175, {FT} Ep.142'}
    (r,) = collect_eval_rows([res])
    assert r.finetune_tile == 'T014'
    assert r.selected_epoch == 175
    assert r.finetune_epoch == 170


def test_subset_filter():
    assert collect_eval_rows([_result(FT, 142)],
                             selected_filenames={'other.gif'}) == []
    assert len(collect_eval_rows([_result(FT, 142)],
                                 selected_filenames={FT})) == 1


def test_ordering_deterministic():
    other = ('x/finetune/AA_T002_rec__from_AA_T000_rec__10/'
             'all_epochs_view0_11-20.gif')
    rows = collect_eval_rows([_result(FT, 150), _result(other, 12)])
    metas = [r.full_metadata for r in rows]
    assert metas == sorted(metas)


def test_build_csv_header_and_row():
    csv_text = build_csv(collect_eval_rows([_result(FT, 142)]))
    lines = csv_text.strip().splitlines()
    assert lines[0] == 'tile,full_metadata_name,selected_finetuning_tile,selected_epoch'
    assert lines[1] == 'T001,NM0029_HT_100nm_T001_0001_rec_,T000,142'


def test_build_sh_uses_batch_eval_with_args():
    sh = build_sh(collect_eval_rows([_result(FT, 142), _result(FT2, 175)]))
    assert sh.startswith('#!/bin/bash')
    # Prompts for the workdir once at the start.
    assert 'read -rp' in sh and 'workdir' in sh
    # One argument-driven sbatch line per row; no sed / eval_manual.sh anymore.
    assert sh.count('sbatch batch_eval.sh') == 2
    assert 'sed' not in sh
    assert 'eval_manual.sh' not in sh
    # Args in order: metadata, epoch, checkpoint_metadata, finetune_epoch, workdir.
    assert ('sbatch batch_eval.sh "NM0029_HT_100nm_T001_0001_rec_" "142" '
            '"NM0029_HT_100nm_T000_0001_rec_" "140" "$workdir"') in sh
    assert ('sbatch batch_eval.sh "NM0029_HT_100nm_T001_0001_rec_" "175" '
            '"NM0029_HT_100nm_T014_0001_rec_" "170" "$workdir"') in sh


def test_command_for_winner_finetune_is_four_args():
    cmd = command_for_winner(f'{FT} Ep.142', metadata_dir='/data/d/denoising/metadata')
    assert cmd == ('submit_eval.sh "/data/d/denoising/metadata/'
                   'NM0029_HT_100nm_T001_0001_rec_.json" "142" '
                   '"NM0029_HT_100nm_T000_0001_rec_" "140"')
    # 3rd arg is a bare checkpoint NAME, not a path
    assert '"NM0029_HT_100nm_T000_0001_rec_"' in cmd


def test_command_for_winner_training_tile_is_two_args():
    # No transfer learning: submit_eval.sh derives checkpoint/finetune itself,
    # so only 2 positional arguments may be emitted (3 would be rejected).
    cmd = command_for_winner(f'{TR} Ep.30', metadata_dir='/data/d/denoising/metadata')
    assert cmd == ('submit_eval.sh "/data/d/denoising/metadata/'
                   'NM0029_HT_100nm_T000_0001_rec_.json" "30"')
    assert cmd.count('"') == 4  # exactly two quoted args


def test_command_for_winner_training_tile_without_tile_token():
    # Live datasets include names with no T<number> token at all.
    gif = ('ls3640/tardi3_tun/corrected_tardi3_tun_HT_real050nm_srs23_0001_rec_/'
           'all_epochs_view0_0-60_raw.gif')
    cmd = command_for_winner(f'{gif} Ep.55', metadata_dir='/m')
    assert cmd == ('submit_eval.sh "/m/corrected_tardi3_tun_HT_real050nm_srs23_0001_rec_.json" '
                   '"55"')


def test_command_for_winner_placeholder_when_no_dir():
    cmd = command_for_winner(f'{TR} Ep.30')
    assert cmd.startswith('submit_eval.sh "<METADATA_DIR>/')
    assert cmd.endswith('.json" "30"')


def test_command_for_winner_strips_trailing_slash():
    a = command_for_winner(f'{TR} Ep.30', metadata_dir='/data/m/')
    b = command_for_winner(f'{TR} Ep.30', metadata_dir='/data/m')
    assert a == b
    assert '//' not in a.split('submit_eval.sh ')[1]


def test_command_for_winner_ignores_non_experiment_dirs():
    # A directory that isn't an experiment at all yields no command.
    assert command_for_winner('some/random/folder/all_epochs_view0_0-9.gif') is None


def test_command_for_winner_unparseable_is_none():
    assert command_for_winner('') is None
    assert command_for_winner('no epoch suffix here.gif') is None
    assert command_for_winner(None) is None


def test_empty_sh_has_no_jobs():
    sh = build_sh([])
    assert sh.startswith('#!/bin/bash')
    assert 'sbatch' not in sh
