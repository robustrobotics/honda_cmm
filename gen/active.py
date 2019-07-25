import sys
import argparse
from util import util
import numpy as np
import argparse
import pybullet as p
from util.setup_pybullet import setup_env, custom_bb_door, custom_bb_slider
from actions import policies
from actions.gripper import Gripper
from gen.generator_busybox import Slider, Door, BusyBox
from collections import namedtuple
from learning.gp_learner import GPLearner
from copy import copy
import operator

Point = namedtuple('Point', 'x z')
Dims = namedtuple('Dims', 'width height')
AttemptedGoal = namedtuple('AttemptedGoal', 'goal competence')

# THIS IS ONLY MADE FOR PRISMATIC POLICIES

class ActivePolicyLearner(object):

    def __init__(self, bb, viz, debug):
        self.bb = bb
        self.mech = self.bb._mechanisms[0]
        self.debug = debug
        self.viz = viz
        self.image_data = setup_env(self.bb, viz, self.debug)
        self.gripper = Gripper(self.bb.bb_id)
        self.max_region = self.get_max_region()
        self.regions = [copy(self.max_region)]
        self.interactions = []

    def explore(self):
        g_max = 10
        while not self.done():
            if self.debug:
                for region in self.regions:
                    region.draw(self.bb)
            region, goal = self.select_goal()
            goal_pos = self.bb.project_onto_backboard([goal[0], 0.0, goal[1]])
            if self.debug:
                util.vis_frame(goal_pos, [0., 0., 0., 1.], lifeTime=0, length=.05)
                for print_region in self.regions:
                    for (other_goal, _) in print_region.attempted_goals:
                        other_goal_pos = self.bb.project_onto_backboard([other_goal[0], 0.0, other_goal[1]])
                        util.vis_frame(other_goal_pos, [0., 0., 0., 1.], lifeTime=0, length=.05)
            handle_pose = self.mech.get_pose_handle_base_world()
            policy, config_goal = policies.Prismatic.get_policy_from_goal(self.bb, self.mech, handle_pose, goal_pos)
            interact_result = self.execute_interaction(policy, config_goal)
            self.interactions += [interact_result]
            competence = self.calc_competence(goal_pos, interact_result)
            region.update(goal, competence)
            self.interactions += [interact_result]
            if len(region.attempted_goals) > g_max:
                new_regions = region.split(self.bb)
                self.regions.remove(region)
                self.regions += new_regions
            self.reset()

    def get_max_region(self):
        # for now regions are just in the x-z plane. will need to move to 3d with doors
        p_bb_base_w = p.getLinkState(self.bb.bb_id,0)[0]
        p_bb_ll_w = Point(p_bb_base_w[0]+self.bb.width/2, p_bb_base_w[2]-self.bb.height/2)
        start_region = Region(Point(p_bb_ll_w[0], p_bb_ll_w[1]),
                                    Dims(self.bb.width, self.bb.height))
        return start_region

    def select_goal(self):
        def interesting_region():
            total_interest = sum([region.interest for region in self.regions])
            probs = [region.interest/total_interest for region in self.regions]
            return np.random.choice(self.regions, p=probs)

        if len(self.regions) > 1:
            mode = np.random.choice([1,2,3],p=[.7,.2,.1])
        else:
            mode = 2

        if mode == 1:
            region = interesting_region()
            goal = region.sample_goal('random')
        elif mode == 2:
            goal = self.max_region.sample_goal('random')
            region = self.get_goal_region(goal)
        elif mode == 3:
            region = interesting_region()
            goal = region.sample_goal('biased')
        return region, goal

    def execute_interaction(self, policy, config_goal):
        pose_handle_world_init = util.Pose(*p.getLinkState(self.bb.bb_id, self.mech.handle_id)[:2])
        pose_handle_base_world = self.mech.get_pose_handle_base_world()
        traj = policy.generate_trajectory(pose_handle_base_world, config_goal, self.debug)
        cumu_motion, net_motion, pose_handle_world_final = \
                self.gripper.execute_trajectory(traj, self.mech, policy.type, self.debug)
        policy_params = policy.get_policy_tuple()
        mechanism_params = self.mech.get_mechanism_tuple()
        return util.Result(policy_params, mechanism_params, net_motion, \
                    cumu_motion, pose_handle_world_init, pose_handle_world_final, \
                    config_goal, self.image_data, None, 1.0)

    def calc_competence(self, goal_pos, result):
        # competence is the (negative) percentage of how much closer you are to
        # the goal than when started
        init_pos = result.pose_joint_world_init.p
        # if gripper flew off, say the handle didn't move
        if result.pose_joint_world_final is None:
            final_pos = init_pos
        else:
            final_pos = result.pose_joint_world_final.p
        dist_to_goal = np.linalg.norm(np.subtract(goal_pos, final_pos))
        init_dist_to_goal = np.linalg.norm(np.subtract(goal_pos, init_pos))
        competence = -np.divide(dist_to_goal, init_dist_to_goal)
        return competence

    def done(self):
        # done when not interested in any regions
        n_interactions = len(self.interactions)
        thresh = 0.005
        overall_interest = sum([region.interest for region in self.regions])
        return overall_interest < thresh and n_interactions > 10

    def reset(self):
        setup_env(self.bb, self.viz, self.debug)
        self.gripper = Gripper(self.bb.bb_id)

    def get_goal_region(self, goal):
        for region in self.regions:
            if region.goal_inside(goal, self.bb):
                return region

class Region(object):
    def __init__(self, coord, dims):
        self.coord = coord # lower left of the region
        self.dims = dims
        self.attempted_goals = [] # in frame of the region coord
        self.interest = float('inf')

    def random_coord(self):
        rand_x = np.random.uniform(self.coord.x, self.coord.x-self.dims.width)
        rand_z = np.random.uniform(self.coord.z, self.coord.z+self.dims.height)
        return rand_x, rand_z

    def sample_goal(self, type):
        R = 0.05
        if type == 'random':
            return Point(*self.random_coord())
        if type == 'biased':
            low_comp_goal, _ = min(self.attempted_goals, key=operator.itemgetter(1))
            r = R*np.sqrt(np.random.uniform())
            theta = np.random.uniform()*2*np.pi
            near_goal = np.add(low_comp_goal, [r*np.cos(theta), r*np.sin(theta)])
            return Point(*near_goal)

    def update(self, goal, competence):
        self.attempted_goals += [(goal, competence)]
        self.update_interest()

    def update_interest(self):
        n_max = 5
        n_goals = len(self.attempted_goals)
        range_max = min(n_max, n_goals)
        self.interest = 0.0
        if n_goals > 1:
            for j in range(-2,-range_max-1,-1):
                self.interest += abs(self.attempted_goals[j+1][1]-self.attempted_goals[j][1])
            self.interest = self.interest/(range_max-1)

    def split(self, bb):
        #print('     splitting region!')
        m = 50
        splits = []
        # randomly generate splits (half xs half zs)
        for dim in range(2):
            for _ in range(m):
                coord = self.random_coord()
                splits += [(dim, coord[dim])]

        quality_values = []
        for (dim, dim_coord) in splits:
            if dim == 0: #x
                coord_right = Point(dim_coord, self.coord.z)
                width_left = self.coord.x - coord_right.x
                width_right = self.dims.width - width_left
                region_0 = Region(self.coord, Dims(width_left, self.dims.height))
                region_1 = Region(coord_right, Dims(width_right, self.dims.height))
            elif dim == 1: #z
                #print('calc z')
                coord_up = Point(self.coord.x, dim_coord)
                height_down = coord_up.z - self.coord.z
                height_up = self.dims.height - height_down
                region_0 = Region(self.coord, Dims(self.dims.width, height_down))
                region_1 = Region(coord_up, Dims(self.dims.width, height_up))

            #region_0.draw(bb)
            #region_1.draw(bb)
            #print(len(self.attempted_goals))
            for goal_comp in self.attempted_goals:
                if region_0.goal_inside(goal_comp[0], bb):
                    region_0.attempted_goals += [goal_comp]
                else:
                    region_1.attempted_goals += [goal_comp]
                #print('break')

            region_0.update_interest()
            region_1.update_interest()
            #print(region_0.interest)
            #print(region_1.interest)
            quality = len(region_0.attempted_goals)*len(region_1.attempted_goals)*\
                        abs(region_0.interest-region_1.interest)
            print(dim, dim_coord, quality)
            quality_values += [((dim, dim_coord, region_0, region_1), quality)]

        # select the split with the highest quality value
        (dim, dim_coord, region_0, region_1), _ = max(quality_values, key=operator.itemgetter(1))
        print('orig goals=', len(self.attempted_goals), 'new0=', len(region_0.attempted_goals),
                    'new1=', len(region_1.attempted_goals))
        return region_0, region_1

    def draw(self, bb, color=[255,255,255], lifeTime=0):
        lift = 0.01
        ll_pos = np.add(bb.project_onto_backboard([self.coord.x, 0.0, self.coord.z]), [0., lift, 0.0])
        lr_pos = np.add(ll_pos, [-self.dims.width,0.0,0.])
        ul_pos = np.add(ll_pos, [0.,0.0,self.dims.height])
        ur_pos = np.add(ll_pos, [-self.dims.width,0.0,self.dims.height])
        p.addUserDebugLine(ll_pos, lr_pos, color, lifeTime=lifeTime)
        p.addUserDebugLine(lr_pos, ur_pos, color, lifeTime=lifeTime)
        p.addUserDebugLine(ur_pos, ul_pos, color, lifeTime=lifeTime)
        p.addUserDebugLine(ul_pos, ll_pos, color, lifeTime=lifeTime)

    def goal_inside(self, goal, bb):
        goal_pos = bb.project_onto_backboard([goal[0], 0.0, goal[1]])
        #util.vis_frame(goal_pos, [0., 0., 0., 1.], lifeTime=.5, length=.1)
        #self.draw(bb, [1,0,0])
        inside = False
        if goal.x < self.coord.x:
            if goal.x > (self.coord.x - self.dims.width):
                if goal.z > self.coord.z:
                    if goal.z < (self.coord.z + self.dims.height):
                        inside = True
        #print(inside)
        return inside

results = []
def generate_dataset(n_samples, viz, debug, urdf_num, max_mech):
    for i in range(n_samples):
        sys.stdout.write("\rProcessing sample %i/%i" % (i+1, n_samples))
        bb = BusyBox.generate_random_busybox(max_mech=max_mech, mech_types=[Slider], urdf_tag=urdf_num, debug=debug)
        active_learner = ActivePolicyLearner(bb, viz, debug)
        active_learner.explore()
        results.extend(active_learner.interactions)
        p.disconnect()
    return results
    print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--viz', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--n-samples', type=int, default=5) # number bbs to generate
    parser.add_argument('--max-mech', type=int, default=1) # mechanisms per bb
    parser.add_argument('--fname', type=str) # give filename
    # if running multiple gens, give then a urdf_num so the correct urdf is read from/written to
    parser.add_argument('--urdf-num', type=int, default=0)
    parser.add_argument('--match-policies', action='store_true')
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()

    try:
        generate_dataset(args.n_samples, args.viz, args.debug, args.urdf_num, args.max_mech)
        if args.fname:
            util.write_to_file(args.fname, results)
    except KeyboardInterrupt:
        # if Ctrl+C write to pickle
        if args.fname:
            util.write_to_file(args.fname, results)
        print('Exiting...')
    except:
        # if crashes write to pickle
        if args.fname:
            util.write_to_file(args.fname, results)

        # for post-mortem debugging since can't run module from command line in pdb.pm() mode
        import traceback, pdb, sys
        traceback.print_exc()
        print('')
        pdb.post_mortem()
        sys.exit(1)
