import csv
from pathlib import Path
import tempfile
import unittest

from run import BOUNDS, DECISIONS, check_front, constraints, selected


class AuditTests(unittest.TestCase):
    def test_selected_row_and_quality_tampering(self):
        row={'selected':'1','quality':'1.2','exposure':'0.5','maximum_receptor':'0.4',
             'fan_power':'1','final_mass_fraction':'0.6'}
        row.update({'decision_'+name:str((lo+hi)/2) for name,(lo,hi) in zip(DECISIONS,BOUNDS)})
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'data.csv'
            def save():
                with path.open('w',newline='') as stream:
                    writer=csv.DictWriter(stream,fieldnames=list(row));writer.writeheader();writer.writerow(row)
            save();selected(path)
            row['quality']='0.1';save()
            with self.assertRaises(ValueError):selected(path)
            row['quality']='1.2';row['selected']='0';save()
            with self.assertRaises(ValueError):selected(path)

    def test_dominance_checks_all_four_objectives(self):
        def point(x):return dict(zip(('exposure','maximum_receptor','fan_power','final_mass_fraction'),map(str,x)))
        check_front([point([1,2,3,4]),point([2,1,3,4])])
        with self.assertRaises(ValueError):check_front([point([1,1,3,4]),point([2,2,3,4])])

    def test_constraints_cannot_be_hidden_by_a_low_score(self):
        self.assertEqual(constraints('MO_SELECTED_CONSTRAINTS [-1, 0, -0.01, -0.0001]','MO_SELECTED')[1],0)
        with self.assertRaises(ValueError):constraints('MO_SELECTED_CONSTRAINTS [-1, 0, 0.001, -0.0001]','MO_SELECTED')
        with self.assertRaises(ValueError):constraints('missing','MO_SELECTED')


if __name__=='__main__':unittest.main(verbosity=2)
