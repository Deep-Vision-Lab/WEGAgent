**The reviewer Agent**

- this agent is responsible for reviewing the *worker* Agents outpus.
to do that, we need to create clear task for each agent, including this.

For each *working* Agents, the reviewer agent should take:
1. The agent task. i.e., the ActionAgent resposnbile for creating action from a description.
2. the mapping of context: output. each working agent should create a response for the reviewer agent of his extraction by giving the context, e.g., in the ActionAgent, for step_x, context: "Clean the spray arm with warm soapy water and a soft brush.", output: Clean. and so on for each step
3. the reviewer agent should take these information, and "review" them by reasoning. foe example, if the action extracted in ActionAgent is "Clean the spray arm with warm soapy water and a soft brush.", and the output action label is: remove, then the reviewer agent should notice this, and notify the actionagent about it to refine it.

**Some** steps may depend on previous steps, e.g., "Repeat the previous operation on the left side...". meaning the current step might be related to previous steps, thus, remember 2-3 previous steps might help reasoning about the current step if vague.
example:
step 13:

* Insert a flat-blade screwdriver into the slot located on the right edge of the ice maker.

* Pry outward with the screwdriver to release the locking tab securing the ice maker to the fridge.

* When the tab releases, pull the right front corner of the ice maker down.

step 14: 

* Repeat the previous operation on the left side of the ice maker to release it.

All agents should preserve short memory (of 2-3 previous step description) in order to reason about actions, parts, and tools.

## Working Agents:
ALL Agents should use full reasoning. think of enhanced pipeline for better context and reasoning.

**Action Agent**
the action agent takes step task name and description, e.g., 
"task_name": "Disable Dishwasher",
"description": "- Turn off the dishwasher.",
and should infer the following:
1. Action describtion, e.g., "Turn off the dishwasher."
2. Action label (verb): Turn off.



Then, it should send a response for the reviewer agent, notifying him that the task is complete, and give him his response in a predifned structure like 
Task definition (e.g., infering action verbs and description from context...)
step_id: Action describtion, Action label



**Tool Agent**
the tool agent should takes step task name and description and tool box, e.g., 
 "task_name": "Remove Left Drawer Slide Screws",
 "description": "- Use a Phillips screwdriver to remove the screws securing both left drawer slides."

 and should infer the tool used in this step. for example, in the aboce information, it should return Phillips screwdriver. 

Then, it should send a response for the reviewer agent, notifying him that the task is complete, and give him his response in a predifned structure like 
Task definition (e.g., `infering tool name from context even if it is not mention directly, e.g., unscrew the bolt around the cover - screwdriver)
step_id: Action describtion, Tool name 

 **Text Part Agent**
 The text part agent takes task name and description, e.g., 
  "task_name": "Remove Left Drawer Slide Screws",
  "description": "- Use a Phillips screwdriver to remove the screws securing both left drawer slides.",

  and should infer all the parts used here. for example, screws, left drawer slides. 
Then, it should send a response for the reviewer agent, notifying him that the task is complete, and give him his response in a predifned structure like 
Task definition (e.g., infering part/component name of the device from context even if it is not mention directly, e.g., unscrew the bolt around the cover - $device cover)
step_id: Action describtion, part/component name

 **Vision Part Agent**
 the vision part agent takes task name and description and the associated image, then build an clear prompt to conclude the objects location wiithen the image. and return the boudning box coordinates as [x1, y1, x2, y2]  where x1, y1 are top left coordinates and x2, y2 bottom right coordinates.



<!-- Then, it should send a response for the reviewer agent, notifying him that the task is complete, and give him his response in a predifned structure like 
Task definition (e.g., infering part/component name of the device from context even if it is not mention directly, e.g., unscrew the bolt around the cover - $device cover)
step_id: Action describtion, part/component name -->