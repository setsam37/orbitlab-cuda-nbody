import math,unittest
from animate import group_frames
def row(step,body,time=None):
    return dict(step=step,time=step*.1 if time is None else time,id=body,x=body,y=0,z=0,vx=0,vy=1,vz=0)
class AnimationTests(unittest.TestCase):
    def test_repeated_ids_across_frames_are_valid(self):
        frames=group_frames([row(0,0),row(0,1),row(1,0),row(1,1)])
        self.assertEqual(len(frames),2);self.assertEqual([r['id'] for r in frames[1]['bodies']],[0,1]);self.assertEqual(frames[1]['time'],.1)
    def test_missing_body_is_rejected(self):
        with self.assertRaises(ValueError):group_frames([row(0,0),row(0,1),row(1,0)])
    def test_duplicate_id_within_frame_is_rejected(self):
        with self.assertRaises(ValueError):group_frames([row(0,0),row(0,0)])
    def test_inconsistent_time_is_rejected(self):
        with self.assertRaises(ValueError):group_frames([row(0,0),row(0,1,.1)])
    def test_nonfinite_position_is_rejected(self):
        bad=row(0,0);bad['x']=math.nan
        with self.assertRaises(ValueError):group_frames([bad])
if __name__=='__main__':unittest.main()
