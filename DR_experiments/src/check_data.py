"""Check the dataset before anything is trained: files, splits, folds and patient separation."""
import sys

import common as C


def main():
    problems = []
    table = C.load_table()
    missing = [i for i in table.image_id if not (C.IMAGES / f'{i}.jpg').is_file()]
    if missing:
        problems.append(f'{len(missing)} photos missing from {C.IMAGES} (first: {missing[:3]})')
    splits = table.groupby('patient_id').split.agg(lambda s: set(s) - {'inadequate'})
    if (splits.map(len) > 1).any():
        problems.append('Some patients appear in more than one of train / val / test.')
    seen = set(table.loc[table.split.isin(['train', 'val']), 'patient_id'])
    if table.loc[table.split == 'inadequate', 'patient_id'].isin(seen).any():
        problems.append('Some inadequate-set patients are in train / val.')
    train = table[table.split == 'train']
    if sorted(train.fold.unique()) != list(range(C.FOLDS)):
        problems.append(f'Training photos must have folds 0..{C.FOLDS - 1}.')
    if (train.groupby('patient_id').fold.nunique() > 1).any():
        problems.append('A patient is spread over several folds.')
    import pandas as pd
    raw = pd.concat([pd.read_csv(f, dtype={'image_id': str, 'patient_id': str}) for f in (C.DIABETIC_CSV, C.INADEQUATE_CSV) if f.is_file()])
    for field, patients in C.clinical_conflicts(raw).items():
        if patients:
            print(f'NOTE: {len(patients)} patient(s) with contradictory {field} in BRSET '
                  f'({raw.patient_id.isin(patients).sum()} photos; patients {patients}); set to missing for these patients.')
    overlap = set(table.loc[table.split == 'inadequate', 'patient_id']) & set(table.loc[table.split == 'test', 'patient_id'])
    print(f'NOTE: {len(overlap)} inadequate-set patients also have photos in the test split (no overlap with train / val); '
          'report the inadequate set as a same-source supplementary low-quality test set.')
    if C.GRADE_CSV.is_file():
        grade = pd.read_csv(C.GRADE_CSV, dtype={'image_id': str}).set_index('image_id').dr_icdr.reindex(table.image_id)
        if grade.isna().any() or ((grade.to_numpy() > 0) != (table[C.TARGET].to_numpy() == 1)).any():
            problems.append('dr_grade.csv does not cover every photo or disagrees with the DR label.')
    else:
        print('NOTE: data/dr_grade.csv not found; the analysis by DR severity will be skipped.')
    for split, part in table.groupby('split'):
        print(f'{split:11s} {len(part):5d} photos  {part.patient_id.nunique():4d} patients  '
              f'{int(part[C.TARGET].sum()):4d} DR-positive ({part[C.TARGET].mean():.1%})')
    print('training folds:', train.groupby('fold').size().to_dict())
    if problems:
        print('\n'.join('PROBLEM: ' + p for p in problems))
        sys.exit(1)
    print('data check passed')


if __name__ == '__main__':
    main()
