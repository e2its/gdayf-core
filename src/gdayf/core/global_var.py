## @package gdayf.core.global_var
# Define all global objects, functions and structs

'''
Copyright (C) e2its - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 *
 * This file is part of gDayF project.
 *
 * Written by Jose L. Sanchez <e2its.es@gmail.com>, 2016-2018
'''


## Function used to set global variable id_user used to propagate user_id to all modules
# @param value User id value
def set_id_user(value):
    global id_user
    id_user = value


## Function used to get global variable id_user used to recover user_id in all modules
# @param model ArMetadata
# @return ArMetadata deepcopy
def get_id_user():
    return id_user


