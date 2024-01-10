'''
Ryan Millett
September 2023

3D Cellular Automata Structures

Inspired by:
Softology - 3D Cellular Automata ("Clouds 1" Algorithm)
https://youtu.be/dQJ5aEsP6Fs?feature=shared&t=61

https://github.com/aaronjolson/Blender-Python-Procedural-Level-Generation

This is an attempt to adapt the aforementioned implementation from 2D to 3D 
using Blender's 'Metaballs' (instead of cube primitives) and to optimize using 
Numpy's vectorization capabilities.

This implementation also adds several layers of post-evaluation analysis, yielding higher
order interpretation functions.
'''

import math
from random import random

import bpy
import bmesh
import numpy as np

INIT_PROB = 0.497
RND_SEED  = 781

ITERATIONS = 27 # Number of iterations to compute up to the present
LOOKAHEAD  = 3   # Number of iterations to compute into the future

X_DIM = Y_DIM = 24
Z_DIM = 36

SIZE = 1
PAD = SIZE * 2
SPACING = SIZE + PAD

np.random.seed(RND_SEED)

def init_map(init_prob=0.5):
    return np.random.rand(X_DIM, Y_DIM, Z_DIM) < init_prob

def apply_rules(old_map):
    count_map = count_neighbors(old_map)
    
    new_map = np.zeros_like(old_map, dtype=bool)
    
    # For cells that are currently alive
    alive_condition = np.logical_and(count_map >= 13, count_map <= 26)
    new_map[old_map] = alive_condition[old_map]

    # For cells that are currently dead
    dead_condition = np.isin(count_map, [13, 14, 17, 18, 19])
    new_map[~old_map] = dead_condition[~old_map]

    return new_map, count_map

def count_neighbors(live_map):
    offsets = [(i, j, k) for i in range(-1, 2) for j in range(-1, 2) for k in range(-1, 2) if (i, j, k) != (0, 0, 0)]
    count = sum(np.roll(np.roll(np.roll(live_map, i, axis=0), j, axis=1), k, axis=2) for i, j, k in offsets)
    return count

def predict_future_states(cell_map, lookahead):
    future_states = [cell_map.copy()]
    for _ in range(lookahead):
        new_map, _ = apply_rules(future_states[-1])
        future_states.append(new_map)
    return future_states

def get_death_iteration(cell_map, x, y, z, lookahead):
    future_states = predict_future_states(cell_map, lookahead)
    for i, future_map in enumerate(future_states):
        if not future_map[x, y, z]:
            return i  # Returns the iteration in which the cell dies
    return False  # if the cell doesn't die within the given window

def create_metaball_object():
    mball = bpy.data.metaballs.new("TempMBall")
    mball_obj = bpy.data.objects.new("TempMBallObj", mball)
    bpy.context.view_layer.active_layer_collection.collection.objects.link(mball_obj)
    return mball

def set_metaball_properties(ele, x, y, z, states):
    state_map = states['states'][ITERATIONS] 
    count_map = states['counts'][ITERATIONS]
    age_map   = states['ages'][ITERATIONS]
    
    is_alive = state_map[x, y, z]
    cell_age = age_map[x, y, z]
    max_age = max([age[x, y, z] for age in states['ages']])
    neighbor_density = count_map[x, y, z] / 26  # 26 is the max number of neighbors
    
#    future_map, future_count = apply_rules(state_map)

    if is_alive:
        size_factor   = np.random.uniform(1, np.interp(cell_age, [1, max_age], [1.62, 6]))
        radius_factor = np.random.uniform(1.33, 2.25) if neighbor_density < 0.57 else np.random.uniform(0.833, 1.667)
        stiffness     = 9 * (1 - (cell_age / max_age)**2) + (1 - neighbor_density)**0.5

        ele.stiffness    = stiffness
        ele.radius       = SIZE * size_factor * radius_factor
        ele.use_negative = cell_age == max_age or not states['states'][ITERATIONS + 1][x, y, z] or neighbor_density > 0.9296
#        ele.use_negative = max([age[x, y, z] for age in states['ages'][ITERATIONS:]]) < LOOKAHEAD and neighbor_density > 0.25
#    else:
#        if future_map[x, y, z]:
#            ele.size = SIZE / 2
#            ele.stiffness = 0.5
#            ele.threshold = 10

def render_cells(states):
    mball = create_metaball_object()
    active_cells = np.transpose(np.nonzero(states['states'][ITERATIONS]))
    
    for x, y, z in active_cells:
        ele = mball.elements.new()
        ele.type = 'BALL'
        ele.co = (x * SPACING + (10 * x / X_DIM),
                  y * SPACING + (10 * y / Y_DIM),
                  z * SPACING + (10 * z / Z_DIM))
        # Set the properties of the metaball element
        set_metaball_properties(ele, x, y, z, states)

def clear_scene():
    if bpy.ops.object.mode_set.poll():
        bpy.ops.object.mode_set(mode='OBJECT')

    # Select and delete all objects
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    # Delete all collections
    for collection in bpy.data.collections:
        bpy.data.collections.remove(collection)

def generate_map():
    states = {
        'states': [],
        'counts': [],
        'ages'  : [],
    }
    
    state_map = init_map(INIT_PROB)
    age_map = np.zeros((X_DIM, Y_DIM, Z_DIM), dtype=int)
    
    for i in range(ITERATIONS + LOOKAHEAD):
        state_map, count_map = apply_rules(state_map)
        age_map[state_map] += 1
        age_map[~state_map] = 0
        
        states['states'].append(state_map.copy())
        states['counts'].append(count_map.copy())
        states['ages'].append(age_map.copy())
    return states


if __name__ == '__main__':
    clear_scene()
    states = generate_map()
    render_cells(states)
