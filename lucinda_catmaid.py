import requests
import re

def clean_text(text):
    return re.sub(r'\W+', '', text)

class Catmaid:
    """
    Represents an instance of CATMAID server
    """

    def __init__(self,url,project_id,api_token):
        self.pid = project_id
        self.url = url
        self.api_token = api_token

    def fetch(self,url,method,data={}):
        """ Fetch data from CATMAID server with HTTP request
        """

        if method == "get":
            result = requests.get(url = self.url + url,
                                    params = data,
                                    headers = {'X-Authorization': 'Token ' + self.api_token})
        if method == "post":
            result = requests.post(url = self.url + url,
                                   data = data,
                                   headers = {'X-Authorization': 'Token ' + self.api_token})
        return result

    def get_skeletons(self):
        """ Return list of skeleton IDs
        """
        return self.fetch(url = str(self.pid) + "/skeletons/",
                           method = "get",
                           data = {"project_id": self.pid}
                         ).json()

    def load_skeleton_names(self,skeletons):
        """ Make dictionary associating skeleton ID and neuron name
        """
        data = {'neuronnames': '1',
                'metaannotations': '0'}

        for i in range(len(skeletons)):
            data["skeleton_ids['" + str(i) + "']"] = skeletons[i]

        source_url = str(self.pid) + "/skeleton/annotationlist"

        return self.fetch(url = source_url,
                          method = "post",
                          data = data).json()['neuronnames']

    def node_overview(self,skid):
        """ Get node overview for skeleton
        """
        url = str(self.pid) + "/skeletons/" + str(skid) + "/node-overview"
        return self.fetch(url = url,
                          method = "get",
                          data = {'project_id': self.pid,
                                  'skeleton_id': skid}).json()