import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# New Antecedent/Consequent objects hold universe variables and membership
# functions

Cholesterol= ctrl.Antecedent(np.arange(0, 500), 'Cholesterol')
Blood_pressure= ctrl.Antecedent(np.arange(0, 200), 'Blood_pressure')
Age= ctrl.Antecedent(np.arange(0, 70), 'Age')
smoking= ctrl.Antecedent(np.arange(0, 1), 'smoking')
BMI= ctrl.Antecedent(np.arange(0, 50), 'BMI')
Diabetes= ctrl.Antecedent(np.arange(0, 400), 'Diabetes')
result = ctrl.Consequent(np.arange(0, 4), 'result')

# Auto-membership function population is possible with .automf(3, 5, or 7)
Cholesterol.automf(3)
Blood_pressure.automf(3)
Age.automf(3)
smoking.automf(3)
BMI.automf(3)
Diabetes.automf(3)

# Custom membership functions can be built interactively with a familiar,
result['healthy'] = fuzz.trapmf(result.universe, [0, 0, 0.4, 1.7])
result['Early_stage'] = fuzz.trimf(result.universe, [1.5, 2, 2.5])
result['Advanced_stage'] = fuzz.trapmf(result.universe, [2.4,3, 4, 4])
Cholesterol.view()
Blood_pressure.view()
Age.view()
smoking.view()
BMI.view()
Diabetes.view()
result.view()

Cholesterol['normal'] = fuzz.trapmf (Cholesterol.universe, [0, 0, 60, 200])
Cholesterol['medium'] = fuzz.trimf (Cholesterol.universe, [190, 210, 250])
Cholesterol['high'] = fuzz.trimf (Cholesterol.universe, [210, 275, 310])
Cholesterol['very high'] = fuzz.trapmf (Cholesterol.universe, [290, 370, 500, 500])
Blood_pressure['normal'] = fuzz.trapmf (Blood_pressure.universe, [0, 0, 79, 138])
Blood_pressure['medium'] = fuzz.trimf (Blood_pressure.universe, [120, 140, 159])
Blood_pressure['high'] = fuzz.trapmf (Blood_pressure.universe, [150, 160, 200, 200])
smoking['false'] = fuzz.trapmf (smoking.universe, [0, 0, 0.5, 0.6])
smoking['true'] = fuzz.trapmf (smoking.universe, [0.5, 0.6, 1, 1])
Age['young'] = fuzz.trapmf (Age.universe, [0, 0, 25, 39])
Age['middle age'] = fuzz.trimf (Age.universe, [32, 40, 45])
Age['old'] = fuzz.trimf (Age.universe, [40, 50, 68])
Age['very old'] = fuzz.trapmf (Age.universe, [52, 60, 70, 70])
BMI['normal'] = fuzz.trapmf (BMI.universe, [0, 0, 12, 26])
BMI['over weight'] = fuzz.trimf (BMI.universe, [25, 27, 32])
BMI['obese'] = fuzz.trapmf (BMI.universe, [30, 42, 50, 50])
Diabetes['normal'] = fuzz.trimf (Diabetes.universe, [0, 75, 155])
Diabetes['diabetic'] = fuzz.trimf (Diabetes.universe, [145, 250, 400])

rule1 = ctrl.Rule(Cholesterol['normal'] & Blood_pressure['normal'] & Age['young'] & smoking['false'] & BMI['normal'] & Diabetes['diabetic'] ,result['healthy'])
rule2 = ctrl.Rule(Cholesterol['high'] & Blood_pressure['high'] & Age['old'] & smoking['false'] & BMI['normal'] & Diabetes['diabetic'] ,result['healthy'])
rule3 = ctrl.Rule(Cholesterol['normal'] & Blood_pressure['high'] & Age['very old'] & smoking['false'] & BMI['obese'] & Diabetes['diabetic'] ,result['healthy'])
rule4 = ctrl.Rule(Cholesterol['high'] & Blood_pressure['high'] & Age['very old' ] & smoking['true'] & BMI['obese'] & Diabetes['normal'] ,result['Early_stage'])
rule5 = ctrl.Rule(Cholesterol['very high'] & Blood_pressure['normal'] & Age['very old'] & BMI['over weight'] & Diabetes['normal'] ,result['Early_stage'])
rule6 = ctrl.Rule(Cholesterol['very high'] & Blood_pressure['high'] & Age['very old'] & smoking['true'] & BMI['obese'] & Diabetes['normal'] ,result['Advanced_stage'])
rule7 = ctrl.Rule(Cholesterol['very high'] & Blood_pressure['high'] & Age['old'] & smoking['false'] & BMI['obese'] & Diabetes['normal'] ,result['Advanced_stage'])
rule8 = ctrl.Rule(Cholesterol['very high'] & Blood_pressure['high'] & Age['very old'] & smoking['false'] & BMI['normal'] & Diabetes['normal'] ,result['Advanced_stage'])

rule1.view()
rule2.view()
rule3.view()
rule4.view()
rule5.view()
rule6.view()
rule7.view()
rule8.view()

res_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4 ,rule5 , rule6, rule7, rule8])
res = ctrl.ControlSystemSimulation(res_ctrl)
res.input['Cholesterol'] = 100
res.input['Blood_pressure'] = 170
res.input['Age'] = 73
res.input['smoking'] = 0
res.input['BMI'] = 37
res.input['Diabetes'] = 219

res.compute()
print("Result = ", res.output['result'])
result.view(sim=res)
