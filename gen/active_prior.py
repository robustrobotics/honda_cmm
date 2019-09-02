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
from copy import copy
import operator
import matplotlib.pyplot as plt
import random
import os
from learning.test_model import get_pred_motions
from copy import copy

Point = namedtuple('Point', 'x z')
Dims = namedtuple('Dims', 'width height')
AttemptedGoal = namedtuple('AttemptedGoal', 'goal competence')

# THIS IS ONLY MADE FOR PRISMATIC POLICIES

# params
g_max = 5  # max samples per region
R = 0.05    # region to sample for low competence
n_max = 5   # maximum number of samples in a region to calc interest
m = 100      # number of samples used to find optimal split
min_region = 0.003
alpha = 0.2

class ActivePolicyLearner(object):

    def __init__(self, bb, viz_sim, debug, viz_plot, all_random):
        self.bb = bb
        self.mech = self.bb._mechanisms[0]
        self.debug = debug
        self.viz_sim = viz_sim
        self.viz_plot = viz_plot
        self.image_data = setup_env(self.bb, self.viz_sim, self.debug)
        self.start_pos = p.getLinkState(self.mech.bb_id, self.mech.handle_id)[0]
        self.gripper = Gripper(self.bb.bb_id)
        self.max_region = self.get_max_region()
        self.regions = [copy(self.max_region)]
        self.interactions = []
        self.all_random = all_random
        #plt.ion()

        if self.viz_plot:
            self.fig, self.ax = plt.subplots()
            self.ax.set_aspect('equal')

            self.reset_plot(self.ax)
            self.made_colorbar = False

    def learn(self, n_int_samples, n_prior_samples, model_path, hdim):
        # if prior model exists then load the model and generate regions from model
        prior_fig = None
        final_fig = None
        if os.path.isfile(model_path):
            model = util.load_model(model_path, hdim=hdim)
            for n in range(n_prior_samples):
                sys.stdout.write("\rProcessing prior sample %i/%i" % (n+1, n_prior_samples))
                if self.debug:
                    for region in self.regions:
                        region.draw(self.bb)
                self.sample(n, 'predict', model)
                if n != n_prior_samples-1:
                    self.reset()
            #if self.viz_plot:
                #prior_fig = copy(self.fig)
        for n in range(n_int_samples):
            sys.stdout.write("\rProcessing interactive sample %i/%i" % (n+1, n_int_samples))
            if self.debug:
                for region in self.regions:
                    region.draw(self.bb)
            self.sample(n, 'interact')
            if n != n_int_samples-1:
                self.reset()
        if self.viz_plot:
            final_fig = copy(self.fig)
        return prior_fig, final_fig

    def sample(self, n, mode, model=None):
        # select goal and region
        prob_explore = np.power(alpha, n)
        choice = np.random.choice(['explore', 'exploit'], p=[prob_explore, 1-prob_explore])
        if choice == 'explore':
            region, goal = self.select_explore_goal()
        else:
            region, goal = self.select_exploit_goal()

        # execute/predict and calculate competence
        goal_pos = self.bb.project_onto_backboard([goal[0], 0.0, goal[1]])
        if self.debug:
            util.vis_frame(goal_pos, [0., 0., 0., 1.], lifeTime=0, length=.05)
            for print_region in self.regions:
                for (other_goal, _) in print_region.attempted_goals:
                    other_goal_pos = self.bb.project_onto_backboard([other_goal[0], 0.0, other_goal[1]])
                    util.vis_frame(other_goal_pos, [0., 0., 0., 1.], lifeTime=0, length=.05)
        handle_pose = self.mech.get_pose_handle_base_world()
        policy, config_goal = policies.Prismatic.get_policy_from_goal(self.bb, self.mech, handle_pose, goal_pos)
        if mode == 'interact':
            interact_result = self.execute_interaction(policy, config_goal)
            self.interactions += [interact_result]
            competence = self.calc_competence(goal_pos, interact_result)
        elif mode == 'predict':
            des_dist = abs(config_goal)
            mech_tuple = self.mech.get_mechanism_tuple()
            policy_tuple = policy.get_policy_tuple()
            result = util.Result(policy_tuple, mech_tuple, 0.0, 0.0,
                                    None, None, config_goal, self.image_data, None, 1.0)
            pred_motion = get_pred_motions([result], model, ret_dataset=False, use_cuda=False)[0]
            # can't have negative competence scores
            competence = max(0.0, np.divide(pred_motion, des_dist))

        # update regions (potentiall split if big enough or have more than g_max goals in region)
        region.update(goal, competence, mode)
        if self.viz_plot:
            self.update_plot((goal, competence), mode=mode)
        if np.multiply(*region.dims) > min_region:
            if len(region.attempted_goals) > g_max:
                new_regions = region.split(self.bb)
                self.regions.remove(region)
                self.regions += new_regions
                if self.viz_plot:
                    self.update_plot(mode=mode)

    def get_max_region(self):
        # for now regions are just in the x-z plane. will need to move to 3d with doors
        p_bb_base_w = p.getLinkState(self.bb.bb_id,0)[0]
        p_bb_ll_w = Point(p_bb_base_w[0]+self.bb.width/2, p_bb_base_w[2]-self.bb.height/2)
        start_region = Region(Point(p_bb_ll_w[0], p_bb_ll_w[1]),
                                    Dims(self.bb.width, self.bb.height))
        return start_region

    def region_interest_probs(self):
        relevant_regions = []
        for region in self.regions:
            if not (len(region.attempted_goals) > g_max and np.multiply(*region.dims) < min_region):
                relevant_regions += [region]
        total_interest = sum([region.interest for region in relevant_regions])
        return tuple([region.interest/total_interest for region in relevant_regions]), tuple(relevant_regions)

    def select_exploit_goal(self):
        probs = []
        for region in self.regions:
            probs += [sum([goal[1] for goal in region.attempted_goals])/len(region.attempted_goals)]
        sum_probs = sum(probs)
        probs = np.divide(probs, sum_probs)
        region = np.random.choice(self.regions, p=probs)
        goal = region.sample_goal('random')
        return region, goal

    def select_explore_goal(self):
        def interesting_region():
            probs, rel_regions = self.region_interest_probs()
            return np.random.choice(rel_regions, p=probs)

        if self.all_random:
            probs = [0., 1., 0.]
        else:
            probs = [.8, .1, .1]

        if len(self.regions) > 1:
            mode = np.random.choice([1,2,3],p=probs)
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
        # competence is how much you moved towards the goal over how far you were
        # initially from the goal
        init_pos = result.pose_joint_world_init.p
        # if gripper flew off, say the handle didn't move
        if result.pose_joint_world_final is None:
            final_pos = init_pos
        else:
            final_pos = result.pose_joint_world_final.p
        init_dist_to_goal = np.linalg.norm(np.subtract([goal_pos[0], goal_pos[2]], [init_pos[0], init_pos[2]]))
        goal_pos_handle = np.subtract(goal_pos, init_pos)
        goal_pos_handle = [goal_pos_handle[0], goal_pos_handle[2]]
        final_pos_handle = np.subtract(final_pos, init_pos)
        final_pos_handle = [final_pos_handle[0], final_pos_handle[2]]
        if np.linalg.norm(final_pos_handle) == 0.0:
            if np.linalg.norm(goal_pos_handle) == 0.0:
                return 1.0
            else:
                return 0.0
        coeff = np.divide(np.dot(final_pos_handle, final_pos_handle), np.linalg.norm(goal_pos_handle)**2)
        motion_proj_handle = np.dot(coeff, goal_pos_handle)
        motion_towards_goal = np.linalg.norm(motion_proj_handle)
        competence = np.divide(motion_towards_goal, init_dist_to_goal)
        return competence

    def reset(self):
        p.resetJointState(self.bb.bb_id, self.mech.handle_id, 0.0)
        self.gripper._set_pose_tip_world(self.gripper.pose_tip_world_reset)

    def get_goal_region(self, goal):
        for region in self.regions:
            if region.goal_inside(goal, self.bb):
                return region

    def plot_mech(self, ax):
        center = [self.start_pos[0], self.start_pos[2]]
        endpoint0 = np.add(center, np.multiply(self.mech.range/2, self.mech.axis))
        endpoint1 = np.add(center, np.multiply(-self.mech.range/2, self.mech.axis))
        ax.plot([-endpoint0[0], -endpoint1[0]], [endpoint0[1], endpoint1[1]], '--r')

    def update_plot(self, element=None, mode=None):
        # is a goal
        if element:
            im = self.ax.scatter([-element[0].x], [element[0].z], c=[element[1]], s=4, vmin=0, vmax=1)
            if not self.made_colorbar:
                self.fig.colorbar(im)
                self.made_colorbar = True
        # just split region, redraw everything
        else:
            # clear figure
            plt.cla()
            self.reset_plot(self.ax)
            interest_probs, rel_regions = self.region_interest_probs()
            for region in self.regions:
                if region in rel_regions:
                    i = rel_regions.index(region)
                    prob = str(interest_probs[i])
                else:
                    # regions that we no longer sample (because small and have g_max samples) are white
                    prob = 'w'
                # draw regions
                corner_coords = region.get_corner_coords()+[region.coord]
                for i in range(4):
                    # flip since axes are reversed in pybullet
                    self.ax.plot(np.multiply(-1,[corner_coords[i].x, corner_coords[i+1].x]),
                                    [corner_coords[i].z, corner_coords[i+1].z], 'k')
                    self.ax.fill_between([-corner_coords[0].x, -corner_coords[1].x],
                                            [corner_coords[0].z, corner_coords[1].z],
                                            [corner_coords[2].z, corner_coords[3].z],
                                            color=prob, alpha=.3)
                # redraw goals
                goals = []
                for element in region.attempted_goals:
                    if mode == element[2]:
                        im = self.ax.scatter([-element[0].x], [element[0].z], c=[element[1]], s=4, vmin=0, vmax=1)
                        if not self.made_colorbar:
                            self.fig.colorbar(im)
                            self.made_colorbar = True
        self.ax.set_title(mode)
        plt.draw()
        plt.pause(0.01)

    def reset_plot(self, ax):
        ax.set_xlim(-self.max_region.coord.x, -1*(self.max_region.coord.x-self.max_region.dims.width))
        ax.set_ylim(self.max_region.coord.z, self.max_region.coord.z+self.max_region.dims.height)
        self.plot_mech(ax)

class Region(object):
    def __init__(self, coord, dims):
        self.coord = coord # lower left of the region
        self.dims = dims
        self.attempted_goals = [] # in frame of the region coord
        self.prior_goals = []
        self.interact_goals = []
        self.interest = float('inf')

    def size(self):
        return np.multiply(*self.dims)

    def random_coord(self):
        rand_x = np.random.uniform(self.coord.x, self.coord.x-self.dims.width)
        rand_z = np.random.uniform(self.coord.z, self.coord.z+self.dims.height)
        return rand_x, rand_z

    def sample_goal(self, type):
        if type == 'random':
            return Point(*self.random_coord())
        if type == 'biased':
            high_comp_goal, _, _ = max(self.attempted_goals, key=operator.itemgetter(1))
            r = R*np.sqrt(np.random.uniform())
            theta = np.random.uniform()*2*np.pi
            near_goal = np.add(high_comp_goal, [r*np.cos(theta), r*np.sin(theta)])

            #plt.plot([-near_goal[0]], [near_goal[1]], 'r.')
            #plt.plot([-high_comp_goal.x], [high_comp_goal.z], 'm.')
            return Point(*near_goal)

    def update(self, goal, competence, mode):
        self.attempted_goals += [(goal, competence, mode)]
        self.update_interest()

    def update_interest(self):
        n_goals = len(self.attempted_goals)
        range_max = min(n_max, n_goals)
        self.interest = 0.0
        if n_goals > 1:
            for j in range(-2,-range_max-1,-1):
                self.interest += abs(self.attempted_goals[j+1][1]-self.attempted_goals[j][1])
            self.interest = self.interest/(range_max-1)

    def split(self, bb):
        splits = []
        ordered_x_goals = sorted(self.attempted_goals, key=lambda goal: goal[0].x)
        ordered_z_goals = sorted(self.attempted_goals, key=lambda goal: goal[0].z)
        for i in range(len(ordered_x_goals)-1):
            diff = abs(ordered_x_goals[i+1][0].x-ordered_x_goals[i][0].x)/2
            x_split = ordered_x_goals[i][0].x + diff
            splits += [(0, x_split)]
        for j in range(len(ordered_z_goals)-1):
            diff = abs(ordered_z_goals[j+1][0].z-ordered_z_goals[j][0].z)/2
            z_split = ordered_z_goals[j][0].z + diff
            splits += [(1, z_split)]

        quality_values = []
        for (dim, dim_coord) in splits:
            if dim == 0: #x
                coord_right = Point(dim_coord, self.coord.z)
                width_left = self.coord.x - coord_right.x
                width_right = self.dims.width - width_left
                region_0 = Region(self.coord, Dims(width_left, self.dims.height))
                region_1 = Region(coord_right, Dims(width_right, self.dims.height))
            elif dim == 1: #z
                coord_up = Point(self.coord.x, dim_coord)
                height_down = coord_up.z - self.coord.z
                height_up = self.dims.height - height_down
                region_0 = Region(self.coord, Dims(self.dims.width, height_down))
                region_1 = Region(coord_up, Dims(self.dims.width, height_up))

            for goal_comp in self.attempted_goals:
                if region_0.goal_inside(goal_comp[0], bb):
                    region_0.attempted_goals += [goal_comp]
                else:
                    region_1.attempted_goals += [goal_comp]

            region_0.update_interest()
            region_1.update_interest()
            quality = len(region_0.attempted_goals)*len(region_1.attempted_goals)*\
                        abs(region_0.interest-region_1.interest)
            quality_values += [((dim, dim_coord, region_0, region_1), quality)]

        # select the split with the highest quality value
        (dim, dim_coord, region_0, region_1), _ = max(quality_values, key=operator.itemgetter(1))
        return region_0, region_1

    def get_corner_coords(self):
        ll_pos = self.coord
        lr_pos = Point(*np.add(ll_pos, [-self.dims.width,0.0]))
        ur_pos = Point(*np.add(ll_pos, [-self.dims.width,self.dims.height]))
        ul_pos = Point(*np.add(ll_pos, [0.0,self.dims.height]))
        return [ll_pos, lr_pos, ur_pos, ul_pos]

    def draw(self, bb, color=[255,255,255], lifeTime=0):
        lift = 0.01
        poses = self.get_corner_coords()+[self.coord]
        y_pos = np.add(bb.project_onto_backboard([self.coord.x, 0.0, self.coord.z]), [0., lift, 0.0])[1]
        for i in range(4):
            pose_i = [poses[i][0], y_pos, poses[i][1]]
            pose_ip1 = [poses[i+1][0], y_pos, poses[i+1][1]]
            p.addUserDebugLine(pose_i, pose_ip1, lifeTime=lifeTime)

    def goal_inside(self, goal, bb):
        goal_pos = bb.project_onto_backboard([goal[0], 0.0, goal[1]])
        inside = False
        if goal.x < self.coord.x:
            if goal.x > (self.coord.x - self.dims.width):
                if goal.z > self.coord.z:
                    if goal.z < (self.coord.z + self.dims.height):
                        inside = True
        return inside

def generate_dataset(n_bbs, n_int_samples, n_prior_samples, viz, debug, urdf_num, max_mech, viz_plot, all_random, bb=None, model_path=None, hdim=16):
    results = []
    prior_figs = []
    final_figs = []
    for i in range(n_bbs):
        if bb is None:
            bb = BusyBox.generate_random_busybox(max_mech=max_mech, mech_types=[Slider], urdf_tag=urdf_num, debug=debug)
        active_learner = ActivePolicyLearner(bb, viz, debug, viz_plot, all_random)
        prior_fig, final_fig = active_learner.learn(n_int_samples, n_prior_samples, model_path, hdim)
        prior_figs += [prior_fig]
        final_figs += [final_fig]
        results.extend(active_learner.interactions)
    print()
    return results, prior_figs, final_figs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--viz', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--n-bbs', type=int, default=5) # number bbs to generate
    parser.add_argument('--n-int-samples', type=int, default=5) # number samples to interact with bb
    parser.add_argument('--n-prior-samples', type=int, default=5) # number samples used to calc prior
    parser.add_argument('--max-mech', type=int, default=1) # mechanisms per bb
    parser.add_argument('--fname', type=str) # give filename if want to save to file
    # if running multiple gens, give then a urdf_num so the correct urdf is read from/written to
    parser.add_argument('--urdf-num', type=int, default=0)
    parser.add_argument('--match-policies', action='store_true') # if want to only use correct policy class on mechanisms
    parser.add_argument('--viz-plot', action='store_true') # if want to run a matplotlib visualization of sampling and competence
    parser.add_argument('--all-random', action='store_true') # if want to only sample randomly
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()

    try:
        results, prior_figs, final_figs = generate_dataset(args.n_bbs, args.n_int_samples, args.n_prior_samples, args.viz, args.debug, args.urdf_num, args.max_mech, args.viz_plot, args.all_random)
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
