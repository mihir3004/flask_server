from flask import Flask, jsonify, redirect, url_for, render_template
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import json

import os

template_dir = os.path.abspath('..')

app = Flask(__name__, template_folder=template_dir)
classifying = None


def fuzzy():
    difflow = ctrl.Antecedent(np.arange(0, 11, 1), 'difflow')
    diffmedium = ctrl.Antecedent(np.arange(0, 11, 1), 'diffmedium')
    diffhard = ctrl.Antecedent(np.arange(0, 11, 1), 'diffhard')
    classification = ctrl.Consequent(np.arange(0, 11, 1), 'classification')
    difflow['poor'] = fuzz.trapmf(difflow.universe, [0, 0, 2, 4])
    difflow['medium'] = fuzz.trapmf(difflow.universe, [2, 4, 6, 8])
    difflow['good'] = fuzz.trapmf(difflow.universe, [6, 8, 10, 10])

    diffmedium['poor'] = fuzz.trapmf(diffmedium.universe, [0, 0, 2, 4])
    diffmedium['medium'] = fuzz.trapmf(diffmedium.universe, [2, 4, 6, 8])
    diffmedium['good'] = fuzz.trapmf(diffmedium.universe, [6, 8, 10, 10])

    diffhard['poor'] = fuzz.trapmf(diffhard.universe, [0, 0, 2, 4])
    diffhard['medium'] = fuzz.trapmf(diffhard.universe, [2, 4, 6, 8])
    diffhard['good'] = fuzz.trapmf(diffhard.universe, [6, 8, 10, 10])
    classification["poor"] = fuzz.trapmf(classification.universe, [0, 0, 2, 4])
    classification["medium"] = fuzz.trapmf(
        classification.universe, [2, 4, 6, 8])
    classification["good"] = fuzz.trapmf(
        classification.universe, [6, 8, 10, 10])
    rule1 = ctrl.Rule((difflow['poor'] & diffmedium['poor'] & diffhard['poor']) |
                      (difflow['poor'] & diffmedium['poor'] & diffhard['medium']) |
                      (difflow['poor'] & diffmedium['poor'] & diffhard['good']) |
                      (difflow['poor'] & diffmedium['good'] & diffhard['good']) |
                      (difflow['medium'] & diffmedium['poor'] & diffhard['good']) |
                      (difflow['poor'] & diffmedium['medium'] & diffhard['good']) |
                      (difflow['poor'] & diffmedium['medium'] & diffhard['poor']) |
                      (difflow['poor'] & diffmedium['medium'] & diffhard['medium']) |
                      (difflow['poor'] & diffmedium['good'] & diffhard['poor']) |
                      (difflow['medium'] & diffmedium['poor'] & diffhard['poor']) |
                      (difflow['medium'] & diffmedium['poor'] & diffhard['medium']), classification['poor'])
    rule2 = ctrl.Rule((difflow['poor'] & diffmedium['medium'] & diffhard['good']) |
                      (difflow['poor'] & diffmedium['good'] & diffhard['medium']) |
                      (difflow['poor'] & diffmedium['good'] & diffhard['good']) |
                      (difflow['medium'] & diffmedium['poor'] & diffhard['good']) |
                      (difflow['medium'] & diffmedium['medium'] & diffhard['poor']) |
                      (difflow['medium'] & diffmedium['medium'] & diffhard['medium']) |
                      (difflow['poor'] & diffmedium['good'] & diffhard['poor']) |
                      (difflow['poor'] & diffmedium['medium'] & diffhard['medium']) |
                      (difflow['medium'] & diffmedium['medium'] & diffhard['good']) |
                      (difflow['medium'] & diffmedium['good'] & diffhard['poor']) |
                      (difflow['good'] & diffmedium['poor'] & diffhard['poor']) |
                      (difflow['good'] & diffmedium['poor'] & diffhard['medium']) |
                      (difflow['good'] & diffmedium['poor'] & diffhard['good']) |
                      (difflow['good'] & diffmedium['medium'] & diffhard['poor']) |
                      (difflow['good'] & diffmedium['medium'] & diffhard['medium']) |
                      (difflow['good'] & diffmedium['good'] & diffhard['poor']), classification['medium'])
    rule3 = ctrl.Rule((difflow['medium'] & diffmedium['good'] & diffhard['medium']) |
                      (difflow['poor'] & diffmedium['medium'] & diffhard['good']) |
                      (difflow['poor'] & diffmedium['good'] & diffhard['good']) |
                      (difflow['medium'] & diffmedium['good'] & diffhard['good']) |
                      (difflow['medium'] & diffmedium['medium'] & diffhard['good']) |
                      (difflow['medium'] & diffmedium['good'] & diffhard['poor']) |
                      (difflow['good'] & diffmedium['poor'] & diffhard['good']) |
                      (difflow['good'] & diffmedium['medium'] & diffhard['medium']) |
                      (difflow['medium'] & diffmedium['medium'] & diffhard['medium']) |
                      (difflow['good'] & diffmedium['medium'] & diffhard['good']) |
                      (difflow['good'] & diffmedium['good'] & diffhard['medium']) |
                      (difflow['good'] & diffmedium['good'] & diffhard['poor']) |
                      (difflow['good'] & diffmedium['good'] & diffhard['good']), classification['good'])
    classifying_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
    global classifying
    classifying = ctrl.ControlSystemSimulation(classifying_ctrl)


fuzzy()


@app.route("/fuzzy/<easy>/<medium>/<hard>")
def fuzzy1(easy, medium, hard):

    global classifying
    diffdict = {
        "difflow": float(easy),
        "diffmedium":  float(medium),
        "diffhard": float(hard),
    }
    classifying.inputs(diffdict)
    classifying.compute()
    result = classifying.output['classification']
    return json.dumps({"result": result})


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=False)
