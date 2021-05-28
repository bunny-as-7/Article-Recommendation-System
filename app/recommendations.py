

"""Recommendation generation module."""

import logging          # for logging information/tasks
import numpy as np      # for numerical computations
import os
import pandas as pd     # for dataframes and data manipulations

import google.auth
import google.cloud.storage as storage      # GCS SDK

logging.basicConfig(level=logging.INFO)

LOCAL_MODEL_PATH = '/tmp'

# these files wii be present in the GCS, created using WALS ML Engine
ROW_MODEL_FILE = 'model/row.npy'
COL_MODEL_FILE = 'model/col.npy'
USER_MODEL_FILE = 'model/user.npy'
ITEM_MODEL_FILE = 'model/item.npy'
USER_ITEM_DATA_FILE = 'data/recommendation_events.csv'


class Recommendations(object):
  """Provide recommendations from a pre-trained collaborative filtering model.
  Args:
    local_model_path: (string) local path to model files
  """

  def __init__(self, local_model_path=LOCAL_MODEL_PATH):
    _, project_id = google.auth.default()
    self._bucket = 'recserve_' + project_id
    self._load_model(local_model_path)


  def _load_model(self, local_model_path):          # will be called only once, during the instantiaion of the class. 
    """Load recommendation model files from GCS.
    Args:
      local_model_path: (string) local path to model files
    """
    # download files from GCS to local storage, code taken from docs
    os.makedirs(os.path.join(local_model_path, 'model'), exist_ok=True)
    os.makedirs(os.path.join(local_model_path, 'data'), exist_ok=True)
    client = storage.Client()
    bucket = client.get_bucket(self._bucket)

    logging.info('Downloading blobs.')

    model_files = [ROW_MODEL_FILE, COL_MODEL_FILE, USER_MODEL_FILE,
                   ITEM_MODEL_FILE, USER_ITEM_DATA_FILE]
    for model_file in model_files:
      blob = bucket.blob(model_file)
      with open(os.path.join(local_model_path, model_file), 'wb') as file_obj:
        blob.download_to_file(file_obj)

    logging.info('Finished downloading blobs.')

    # load npy arrays for user/item factors and user/item maps
    self.user_factor = np.load(os.path.join(local_model_path, ROW_MODEL_FILE))
    self.item_factor = np.load(os.path.join(local_model_path, COL_MODEL_FILE))
    self.user_map = np.load(os.path.join(local_model_path, USER_MODEL_FILE))        # mapping of transformed user_id with its actual id
    self.item_map = np.load(os.path.join(local_model_path, ITEM_MODEL_FILE))        # mapping of transformed item_id with its actual id

    logging.info('Finished loading arrays.')

    # load user_item history into pandas dataframe
    views_df = pd.read_csv(os.path.join(local_model_path,
                                        USER_ITEM_DATA_FILE), sep=',', header=0)
    self.user_items = views_df.groupby('clientId')

    logging.info('Finished loading model.')


  def get_recommendations(self, user_id, num_recs):

    # map user id into ratings matrix user index      
    user_idx = np.searchsorted(self.user_map, user_id)

    if user_idx:
      # get already viewed items from views dataframe, as we don't need to include these in recommendations.
      already_rated = self.user_items.get_group(user_id).contentId
      already_rated_idx = [np.searchsorted(self.item_map, i)
                           for i in already_rated]

      # generate list of recommended article indexes from model
      recommendations = generate_recommendations(user_idx, already_rated_idx,
                                                 self.user_factor,
                                                 self.item_factor,
                                                 num_recs)

      # map article indexes back to article ids
      article_recommendations = [self.item_map[i] for i in recommendations]       # map transformed item id with the actual id

    return article_recommendations


def generate_recommendations(user_idx, user_rated, row_factor, col_factor, k):
                          #                       user_factor item_factor num_recs

  # bounds checking for args
  assert (row_factor.shape[0] - len(user_rated)) >= k

  # retrieve user factor
  user_f = row_factor[user_idx]

  # dot product of item factors with user factor gives predicted ratings
  pred_ratings = col_factor.dot(user_f)

  # find candidate recommended item indexes sorted by predicted rating
  k_r = k + len(user_rated)
  candidate_items = np.argsort(pred_ratings)[-k_r:]

  # remove previously rated items and take top k
  recommended_items = [i for i in candidate_items if i not in user_rated]
  recommended_items = recommended_items[-k:]

  # flip to sort highest rated first
  recommended_items.reverse()

  return recommended_items
