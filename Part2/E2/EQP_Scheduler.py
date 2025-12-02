from enum import Enum
import numpy as np
import pygame
import math

class VTRAction(Enum):
    NO_ACTION = 0
    ENTRY_TO_LEFT_ARM = 1
    ENTRY_TO_RIGHT_ARM = 2
    LEFT_ARM_TO_CHAMBER_1 = 3
    RIGHT_ARM_TO_CHAMBER_1 = 4
    LEFT_ARM_TO_CHAMBER_2 = 5
    RIGHT_ARM_TO_CHAMBER_2 = 6
    LEFT_ARM_TO_CHAMBER_3 = 7
    RIGHT_ARM_TO_CHAMBER_3 = 8
    LEFT_ARM_TO_CHAMBER_4 = 9
    RIGHT_ARM_TO_CHAMBER_4 = 10
    LEFT_ARM_TO_CHAMBER_5 = 11
    RIGHT_ARM_TO_CHAMBER_5 = 12
    LEFT_ARM_TO_CHAMBER_6 = 13
    RIGHT_ARM_TO_CHAMBER_6 = 14
    CHAMBER_1_TO_LEFT_ARM = 15
    CHAMBER_1_TO_RIGHT_ARM = 16
    CHAMBER_2_TO_LEFT_ARM = 17
    CHAMBER_2_TO_RIGHT_ARM = 18
    CHAMBER_3_TO_LEFT_ARM = 19
    CHAMBER_3_TO_RIGHT_ARM = 20
    CHAMBER_4_TO_LEFT_ARM = 21
    CHAMBER_4_TO_RIGHT_ARM = 22
    CHAMBER_5_TO_LEFT_ARM = 23
    CHAMBER_5_TO_RIGHT_ARM = 24
    CHAMBER_6_TO_LEFT_ARM = 25
    CHAMBER_6_TO_RIGHT_ARM = 26
    LEFT_ARM_TO_EXIT = 27
    RIGHT_ARM_TO_EXIT = 28

class ATRAction(Enum):
    NO_ACTION = 0
    LP_TO_LEFT_ARM = 1
    LP_TO_RIGHT_ARM = 2
    LEFT_ARM_TO_ALIGNER = 3
    RIGHT_ARM_TO_ALIGNER = 4
    ALIGNER_TO_LEFT_ARM = 5
    ALIGNER_TO_RIGHT_ARM = 6
    LEFT_ARM_TO_ENTRY = 7
    RIGHT_ARM_TO_ENTRY = 8
    EXIT_TO_LEFT_ARM = 9
    EXIT_TO_RIGHT_ARM = 10
    LEFT_ARM_TO_LP = 11
    RIGHT_ARM_TO_LP = 12

class WaferProcessState(Enum):
    BEFORE_PROCESS = 0
    ALIGNED = 1
    PROCESSED = 2

class ComponentType(Enum):
    ENTRY = 0
    EXIT = 1
    VTR_LEFT_ARM = 2
    VTR_RIGHT_ARM = 3
    CHAMBER = 4
    ATR_LEFT_ARM = 5
    ATR_RIGHT_ARM = 6
    ALIGNER = 7
    LP_IN = 8
    LP_OUT = 9

class ATRComponentLocation(Enum):
    LP_IN = 0
    ENTRY = 1
    EXIT = 2
    ALIGNER = 3
    LP_OUT = 4

class VTRComponentLocation(Enum):
    ENTRY = 0
    CHAMBER_1 = 1
    CHAMBER_2 = 2
    CHAMBER_3 = 3
    CHAMBER_4 = 4
    CHAMBER_5 = 5
    CHAMBER_6 = 6
    EXIT = 7

class Wafer:
    def __init__(self, wafer_id):
        self.id = wafer_id
        self.process_state = WaferProcessState.BEFORE_PROCESS

class Component:
    def __init__(self, component_type, index=0):
        self.type = component_type
        self.index = index
        self.holding_wafer = None
        self.process_time = 0
        self.cumulative_process_time = 0
        self.total_cumulative_process_time = 0
        self.processed_wait_time = 0  # 새로운 속성
        self.vac_or_atm = 0 #0 = vac, 1 = atm
        self.pending_action = None
        self.location = 0

class EQP_Scheduler:

    def __init__(self, MAX_WAFER_NO, VTR_EXECUTION_TIME, CMB_MIN_EXECUTION_TIME, CMB_MAX_EXECUTION_TIME, 
                 VAC_ATM_CHANGE_TIME, CMD_CLEANING_LIMIT, CMD_CLEANING_TIME, ATR_EXECUTION_TIME, ALIENER_EXECUTION_TIME, LP_IN_OUT_PROCESS_TIME):
        self.VTR_ExecutionTime = VTR_EXECUTION_TIME
        self.CMB_MinExecutionTime = CMB_MIN_EXECUTION_TIME
        self.CMB_MaxExecutionTime = CMB_MAX_EXECUTION_TIME
        self.ENTRY_EXIT_PROCESS_TIME = VAC_ATM_CHANGE_TIME
        self.CMD_CLEANING_LIMIT = CMD_CLEANING_LIMIT
        self.CMD_CLEANING_TIME = CMD_CLEANING_TIME

        self.ATR_ExecutionTime = ATR_EXECUTION_TIME
        self.ALIENER_ExecutionTime = ALIENER_EXECUTION_TIME
        self.LP_IN_OUT_PROCESS_TIME = LP_IN_OUT_PROCESS_TIME

        self.components = {
            ComponentType.ENTRY: Component(ComponentType.ENTRY),
            ComponentType.EXIT: Component(ComponentType.EXIT),
            ComponentType.VTR_LEFT_ARM: Component(ComponentType.VTR_LEFT_ARM),
            ComponentType.VTR_RIGHT_ARM: Component(ComponentType.VTR_RIGHT_ARM),
            ComponentType.CHAMBER: [Component(ComponentType.CHAMBER, i) for i in range(6)],
            ComponentType.ATR_LEFT_ARM: Component(ComponentType.ATR_LEFT_ARM),
            ComponentType.ATR_RIGHT_ARM: Component(ComponentType.ATR_RIGHT_ARM),
            ComponentType.ALIGNER: Component(ComponentType.ALIGNER),
            ComponentType.LP_IN: Component(ComponentType.LP_IN),
            ComponentType.LP_OUT: Component(ComponentType.LP_OUT)
        }

        
        self.wafers = []
        self.next_wafer_id = 1

        self.max_wafer_no = MAX_WAFER_NO
        self.processed_wafer_count = 0 

        self.VTR_LocationCount = 8
        self.ATR_LocationCount = 5

        self.reset()

    def reset(self, seed=None):
        pygame.init()
        self.screen = pygame.display.set_mode((1000, 800))
        self.clock = pygame.time.Clock()
        pygame.display.set_caption('CustomEnv Visualization')

        self.current_eqp_step = 0
        self.total_processed_wafers = 0
        self.processed_wafer_count = 0 
        self.vtr_last_action = VTRAction.NO_ACTION
        self.atr_last_action = ATRAction.NO_ACTION
        np.random.seed(seed)

        for component_type, component_list in self.components.items():
            if component_type == ComponentType.CHAMBER:
                for component in component_list:
                    self._reset_component(component, vac_or_atm=0)
            elif component_type == ComponentType.VTR_LEFT_ARM or component_type == ComponentType.VTR_RIGHT_ARM:
                self._reset_component(component_list, vac_or_atm=0)
            elif component_type == ComponentType.EXIT:
                self._reset_component(component_list, vac_or_atm=0)
            else:
                if isinstance(component_list, list):
                    for component in component_list:
                        self._reset_component(component, vac_or_atm=1)
                else:
                    self._reset_component(component_list, vac_or_atm=1)

        self.wafers = [Wafer(i) for i in range(self.max_wafer_no)]
        self.next_wafer_id = 1

        # Set the first wafer in the LP_IN component
        first_wafer = self.wafers[0]
        self.components[ComponentType.LP_IN].holding_wafer = first_wafer

    def _reset_component(self, component, vac_or_atm):
        component.holding_wafer = None
        component.process_time = 0
        component.cumulative_process_time = 0
        component.total_cumulative_process_time = 0
        component.processed_wait_time = 0
        component.vac_or_atm = vac_or_atm
        component.pending_action = None
        if component.type in [ComponentType.ATR_LEFT_ARM, ComponentType.ATR_RIGHT_ARM]:
            component.location = ATRComponentLocation(0)
        elif component.type in [ComponentType.VTR_LEFT_ARM, ComponentType.VTR_RIGHT_ARM]:
            component.location = VTRComponentLocation(0)
        else:
            component.location = 0


    def getVTRExeTime(self, component, next_location, update_location = True):
        current_location = component.location

        # Ensure both locations are VTRComponentLocation
        if not isinstance(current_location, VTRComponentLocation):
            raise ValueError(f"Current location {current_location} is not a valid VTRComponentLocation")
        if not isinstance(next_location, VTRComponentLocation):
            raise ValueError(f"Next location {next_location} is not a valid VTRComponentLocation")

        # Calculate the shortest distance
        distance = min((next_location.value - current_location.value) % self.VTR_LocationCount,
                    (current_location.value - next_location.value) % self.VTR_LocationCount) + 1

        if update_location:
            # Update ATR location
            component.location = next_location

            # Calculate opposite location for the other arm
            opposite_location = VTRComponentLocation((next_location.value + 4) % self.VTR_LocationCount)
        
            if component.type == ComponentType.VTR_LEFT_ARM:
                self.components[ComponentType.VTR_RIGHT_ARM].location = opposite_location
            elif component.type == ComponentType.VTR_RIGHT_ARM:
                self.components[ComponentType.VTR_LEFT_ARM].location = opposite_location


        # Calculate and return execution time
        return distance * self.VTR_ExecutionTime

    def getATRExeTime(self, component, next_location, update_location = True):
        current_location = component.location

        # Ensure both locations are ATRComponentLocation
        if not isinstance(current_location, ATRComponentLocation):
            raise ValueError(f"Current location {current_location} is not a valid ATRComponentLocation")
        if not isinstance(next_location, ATRComponentLocation):
            raise ValueError(f"Next location {next_location} is not a valid ATRComponentLocation")

        # Calculate the shortest distance
        distance = min((next_location.value - current_location.value) % self.ATR_LocationCount,
                    (current_location.value - next_location.value) % self.ATR_LocationCount) + 1

        if update_location:
            # Update ATR location
            component.location = next_location

            # Calculate opposite location for the other arm
            opposite_location = (next_location.value + 2) % self.ATR_LocationCount
            opposite_location = ATRComponentLocation(opposite_location)

            if component.type == ComponentType.ATR_LEFT_ARM:
                self.components[ComponentType.ATR_RIGHT_ARM].location = opposite_location
            elif component.type == ComponentType.ATR_RIGHT_ARM:
                self.components[ComponentType.ATR_LEFT_ARM].location = opposite_location

        # Calculate and return execution time
        return distance * self.ATR_ExecutionTime

    def getALIGNERExeTime(self):
        return self.ALIENER_ExecutionTime

    def getCMBExeTime(self):
        return np.random.randint(self.CMB_MinExecutionTime, self.CMB_MaxExecutionTime)

    def getATRAvailable(self):
        atr_left_arm = self.components[ComponentType.ATR_LEFT_ARM]
        atr_right_arm = self.components[ComponentType.ATR_RIGHT_ARM]

        if atr_left_arm.process_time == 0 and atr_right_arm.process_time == 0:
            return True
        else:
            return False

    def getVTRAvailable(self):
        vtr_left_arm = self.components[ComponentType.VTR_LEFT_ARM]
        vtr_right_arm = self.components[ComponentType.VTR_RIGHT_ARM]

        if vtr_left_arm.process_time == 0 and vtr_right_arm.process_time == 0:
            return True
        else:
            return False
    
    def getComponentAvailable(self, component):
        if component.process_time == 0 and component.pending_action is None:
            return True
        else:
            return False
        
    

    def check_atr_action_validity(self, atr_action:ATRAction):
        entry = self.components[ComponentType.ENTRY]
        exit = self.components[ComponentType.EXIT]
        atr_left_arm = self.components[ComponentType.ATR_LEFT_ARM]
        atr_right_arm = self.components[ComponentType.ATR_RIGHT_ARM]
        aligner = self.components[ComponentType.ALIGNER]
        lp_in = self.components[ComponentType.LP_IN]
        lp_out = self.components[ComponentType.LP_OUT]

        both_arms_empty = atr_left_arm.holding_wafer is None and atr_right_arm.holding_wafer is None

        if atr_action == ATRAction.NO_ACTION:
            return True
        elif atr_action == ATRAction.LP_TO_LEFT_ARM:
            if not both_arms_empty:
                return False
            
            ret = self.getATRAvailable() and lp_in.holding_wafer is not None and atr_left_arm.holding_wafer is None and self.getComponentAvailable(lp_in)
            return ret
        elif atr_action == ATRAction.LP_TO_RIGHT_ARM:
            if not both_arms_empty:
                return False

            ret = self.getATRAvailable() and lp_in.holding_wafer is not None and atr_right_arm.holding_wafer is None and self.getComponentAvailable(lp_in)
            return ret
        elif atr_action == ATRAction.LEFT_ARM_TO_ALIGNER:
            return (self.getATRAvailable() and 
                    self.getComponentAvailable(aligner) and 
                    atr_left_arm.holding_wafer is not None and 
                    aligner.holding_wafer is None and
                    atr_left_arm.holding_wafer.process_state == WaferProcessState.BEFORE_PROCESS)       
        elif atr_action == ATRAction.RIGHT_ARM_TO_ALIGNER:
            return (self.getATRAvailable() and 
                    self.getComponentAvailable(aligner) and 
                    atr_right_arm.holding_wafer is not None and 
                    aligner.holding_wafer is None and 
                    atr_right_arm.holding_wafer.process_state == WaferProcessState.BEFORE_PROCESS)
        elif atr_action == ATRAction.ALIGNER_TO_LEFT_ARM:
            return (self.getComponentAvailable(aligner) and 
                    self.getATRAvailable() and 
                    aligner.holding_wafer is not None and 
                    aligner.holding_wafer.process_state == WaferProcessState.ALIGNED and
                    atr_left_arm.holding_wafer is None and
                    # VTR/ATR deadlock prevention
                    entry.holding_wafer is None)
        elif atr_action == ATRAction.ALIGNER_TO_RIGHT_ARM:
            return (self.getComponentAvailable(aligner) and 
                    self.getATRAvailable() and 
                    aligner.holding_wafer is not None and
                    aligner.holding_wafer.process_state == WaferProcessState.ALIGNED and 
                    atr_right_arm.holding_wafer is None and
                    # VTR/ATR deadlock prevention
                    entry.holding_wafer is None)
        elif atr_action == ATRAction.LEFT_ARM_TO_ENTRY:
            return (entry.holding_wafer is None and 
                    entry.vac_or_atm == 1 and
                    self.getComponentAvailable(entry) and
                    self.getATRAvailable() and 
                    atr_left_arm.holding_wafer is not None and
                    atr_left_arm.holding_wafer.process_state == WaferProcessState.ALIGNED)
        elif atr_action == ATRAction.RIGHT_ARM_TO_ENTRY:
            return (entry.holding_wafer is None and
                    entry.vac_or_atm == 1 and
                    self.getComponentAvailable(entry) and
                    self.getATRAvailable() and
                    atr_right_arm.holding_wafer is not None and
                    atr_right_arm.holding_wafer.process_state == WaferProcessState.ALIGNED)
        elif atr_action == ATRAction.EXIT_TO_LEFT_ARM:
            return (exit.holding_wafer is not None and
                    exit.vac_or_atm == 1 and
                    self.getComponentAvailable(exit) and
                    exit.holding_wafer.process_state == WaferProcessState.PROCESSED and
                    self.getATRAvailable() and 
                    atr_left_arm.holding_wafer is None)
        elif atr_action == ATRAction.EXIT_TO_RIGHT_ARM:
            return (exit.holding_wafer is not None and
                    exit.vac_or_atm == 1 and
                    self.getComponentAvailable(exit) and
                    exit.holding_wafer.process_state == WaferProcessState.PROCESSED and
                    self.getATRAvailable() and
                    atr_right_arm.holding_wafer is None)
        elif atr_action == ATRAction.LEFT_ARM_TO_LP:
            return (self.getATRAvailable() and 
                    lp_out.holding_wafer is None and 
                    atr_left_arm.holding_wafer is not None and
                    atr_left_arm.holding_wafer.process_state == WaferProcessState.PROCESSED and 
                    self.getComponentAvailable(lp_out))
        elif atr_action == ATRAction.RIGHT_ARM_TO_LP:
            return (self.getATRAvailable() and 
                    lp_out.holding_wafer is None and 
                    atr_right_arm.holding_wafer is not None and
                    atr_right_arm.holding_wafer.process_state == WaferProcessState.PROCESSED and
                    self.getComponentAvailable(lp_out)) 
        

    def check_vtr_action_validity(self, vtr_action: VTRAction):
        entry = self.components[ComponentType.ENTRY]
        exit = self.components[ComponentType.EXIT]
        vtr_left_arm = self.components[ComponentType.VTR_LEFT_ARM]
        vtr_right_arm = self.components[ComponentType.VTR_RIGHT_ARM]
        chambers = self.components[ComponentType.CHAMBER]

        if vtr_action == VTRAction.NO_ACTION:
            return True
        elif vtr_action == VTRAction.ENTRY_TO_LEFT_ARM or vtr_action == VTRAction.ENTRY_TO_RIGHT_ARM:
            chambers_have_wafer = any(chamber.holding_wafer is not None for chamber in chambers)
            both_arms_empty = vtr_left_arm.holding_wafer is None and vtr_right_arm.holding_wafer is None
            
            if chambers_have_wafer:
                return (entry.holding_wafer is not None and 
                        entry.vac_or_atm == 0 and
                        self.getVTRAvailable() and
                        both_arms_empty and self.getComponentAvailable(entry))
            else:
                if vtr_action == VTRAction.ENTRY_TO_LEFT_ARM:
                    return (self.getVTRAvailable() and 
                            entry.holding_wafer is not None and 
                            entry.vac_or_atm == 0 and
                            self.getComponentAvailable(entry) and
                            vtr_left_arm.holding_wafer is None)
                else:  # ENTRY_TO_RIGHT_ARM
                    return (self.getVTRAvailable() and 
                            entry.holding_wafer is not None and 
                            entry.vac_or_atm == 0 and
                            self.getComponentAvailable(entry) and
                            vtr_right_arm.holding_wafer is None)
        elif vtr_action.name.startswith("LEFT_ARM_TO_CHAMBER"):
            chamber_index = int(vtr_action.name.split('_')[-1]) - 1
            return (self.getVTRAvailable() and 
                    self.getComponentAvailable(chambers[chamber_index]) and
                    vtr_left_arm.holding_wafer is not None and 
                    chambers[chamber_index].holding_wafer is None and 
                    vtr_left_arm.holding_wafer.process_state == WaferProcessState.ALIGNED)
        elif vtr_action.name.startswith("RIGHT_ARM_TO_CHAMBER"):
            chamber_index = int(vtr_action.name.split('_')[-1]) - 1
            return (self.getVTRAvailable() and 
                    self.getComponentAvailable(chambers[chamber_index]) and
                    vtr_right_arm.holding_wafer is not None and 
                    chambers[chamber_index].holding_wafer is None and 
                    vtr_right_arm.holding_wafer.process_state == WaferProcessState.ALIGNED)
        elif vtr_action.name.startswith("CHAMBER"):
            chamber_index = int(vtr_action.name.split('_')[1]) - 1
            if "LEFT_ARM" in vtr_action.name:
                return (self.getVTRAvailable() and 
                        self.getComponentAvailable(chambers[chamber_index]) and
                        vtr_left_arm.holding_wafer is None and 
                        chambers[chamber_index].holding_wafer is not None and 
                        chambers[chamber_index].holding_wafer.process_state == WaferProcessState.PROCESSED)
            elif "RIGHT_ARM" in vtr_action.name:
                return (self.getVTRAvailable() and 
                        self.getComponentAvailable(chambers[chamber_index]) and
                        vtr_right_arm.holding_wafer is None and 
                        chambers[chamber_index].holding_wafer is not None and 
                        chambers[chamber_index].holding_wafer.process_state == WaferProcessState.PROCESSED)
        elif vtr_action == VTRAction.LEFT_ARM_TO_EXIT:
            return (self.getVTRAvailable() and 
                    vtr_left_arm.holding_wafer is not None and 
                    exit.holding_wafer is None and 
                    exit.vac_or_atm == 0 and
                    self.getComponentAvailable(exit) and
                    vtr_left_arm.holding_wafer.process_state == WaferProcessState.PROCESSED)
        elif vtr_action == VTRAction.RIGHT_ARM_TO_EXIT:
            return (self.getVTRAvailable() and 
                    vtr_right_arm.holding_wafer is not None and 
                    exit.holding_wafer is None and 
                    exit.vac_or_atm == 0 and
                    self.getComponentAvailable(exit) and
                    vtr_right_arm.holding_wafer.process_state == WaferProcessState.PROCESSED)

        return False
    
    def _update_component_times(self):
        for component_type, component in self.components.items():
            if component_type == ComponentType.CHAMBER:
                # Process Modules (Chambers) are handled separately
                for chamber in component:
                    if chamber.process_time > 0:
                        chamber.process_time -= 1
                        if chamber.process_time == 0 and chamber.holding_wafer is not None:
                            chamber.holding_wafer.process_state = WaferProcessState.PROCESSED

            else:
                # All other components
                if component.process_time > 0:
                    component.process_time -= 1
                    if component.process_time == 0:
                        self._complete_pending_action(component)


    def _complete_pending_action(self, component):
        
        if component.type == ComponentType.ALIGNER:
                aligner = self.components[ComponentType.ALIGNER]
                if aligner.process_time == 0 and aligner.holding_wafer is not None:
                    aligner.holding_wafer.process_state = WaferProcessState.ALIGNED
        elif component.type == ComponentType.LP_IN:
                lp_in = self.components[ComponentType.LP_IN]
                if lp_in.process_time == 0 and lp_in.holding_wafer is None and self.next_wafer_id < len(self.wafers):
                    next_wafer = self.wafers[self.next_wafer_id]
                    lp_in.holding_wafer = next_wafer
                    self.next_wafer_id += 1
        else:
            if component.pending_action:
                if component.type in [ComponentType.ATR_LEFT_ARM, ComponentType.ATR_RIGHT_ARM]:
                    self._complete_atr_action(component)
                elif component.type in [ComponentType.VTR_LEFT_ARM, ComponentType.VTR_RIGHT_ARM]:
                    self._complete_vtr_action(component)
                elif component.type in [ComponentType.ENTRY, ComponentType.EXIT]:
                    self._complete_ll_action(component)
            
    def _complete_ll_action(self, component):

        if component.process_time == 0 and isinstance(component.pending_action, ATRAction):
            component.vac_or_atm = 0
        elif component.process_time == 0 and isinstance(component.pending_action, VTRAction):
            component.vac_or_atm = 1
        
        component.pending_action = None

    def _complete_atr_action(self, component):
        entry = self.components[ComponentType.ENTRY]
        exit = self.components[ComponentType.EXIT]
        atr_left_arm = self.components[ComponentType.ATR_LEFT_ARM]
        atr_right_arm = self.components[ComponentType.ATR_RIGHT_ARM]
        aligner = self.components[ComponentType.ALIGNER]
        lp_in = self.components[ComponentType.LP_IN]
        lp_out = self.components[ComponentType.LP_OUT]
    
        atr_action = component.pending_action

        if atr_action == ATRAction.LP_TO_LEFT_ARM:
            atr_left_arm.holding_wafer = lp_in.holding_wafer
            lp_in.holding_wafer = None
            lp_in.process_time = self.LP_IN_OUT_PROCESS_TIME
            lp_in.pending_action = None
            atr_left_arm.pending_action = None

        elif atr_action == ATRAction.LP_TO_RIGHT_ARM:
            atr_right_arm.holding_wafer = lp_in.holding_wafer
            lp_in.holding_wafer = None
            lp_in.process_time = self.LP_IN_OUT_PROCESS_TIME
            lp_in.pending_action = None
            atr_right_arm.pending_action = None

        elif atr_action == ATRAction.LEFT_ARM_TO_ALIGNER:
            aligner.holding_wafer = atr_left_arm.holding_wafer
            aligner.process_time = self.getALIGNERExeTime()
            atr_left_arm.holding_wafer = None
            aligner.pending_action = None
            atr_left_arm.pending_action = None

        elif atr_action == ATRAction.RIGHT_ARM_TO_ALIGNER:
            aligner.holding_wafer = atr_right_arm.holding_wafer
            aligner.process_time = self.getALIGNERExeTime()
            atr_right_arm.holding_wafer = None
            aligner.pending_action = None
            atr_right_arm.pending_action = None

        elif atr_action == ATRAction.ALIGNER_TO_LEFT_ARM:
            atr_left_arm.holding_wafer = aligner.holding_wafer
            aligner.holding_wafer = None
            aligner.pending_action = None
            atr_left_arm.pending_action = None

        elif atr_action == ATRAction.ALIGNER_TO_RIGHT_ARM:
            atr_right_arm.holding_wafer = aligner.holding_wafer
            aligner.holding_wafer = None
            aligner.pending_action = None
            atr_right_arm.pending_action = None

        elif atr_action == ATRAction.LEFT_ARM_TO_ENTRY:
            entry.holding_wafer = atr_left_arm.holding_wafer
            entry.process_time = self.ENTRY_EXIT_PROCESS_TIME
            atr_left_arm.holding_wafer = None
            atr_left_arm.pending_action = None

        elif atr_action == ATRAction.RIGHT_ARM_TO_ENTRY:
            entry.holding_wafer = atr_right_arm.holding_wafer
            entry.process_time = self.ENTRY_EXIT_PROCESS_TIME
            atr_right_arm.holding_wafer = None
            atr_right_arm.pending_action = None

        elif atr_action == ATRAction.EXIT_TO_LEFT_ARM:
            atr_left_arm.holding_wafer = exit.holding_wafer
            exit.holding_wafer = None
            exit.process_time = self.ENTRY_EXIT_PROCESS_TIME
            atr_left_arm.pending_action = None

        elif atr_action == ATRAction.EXIT_TO_RIGHT_ARM:
            atr_right_arm.holding_wafer = exit.holding_wafer
            exit.holding_wafer = None
            exit.process_time = self.ENTRY_EXIT_PROCESS_TIME
            atr_right_arm.pending_action = None

        elif atr_action == ATRAction.LEFT_ARM_TO_LP:
            lp_out.holding_wafer = atr_left_arm.holding_wafer
            lp_out.process_time = self.LP_IN_OUT_PROCESS_TIME
            atr_left_arm.holding_wafer = None
            lp_out.pending_action = None
            atr_left_arm.pending_action = None

        elif atr_action == ATRAction.RIGHT_ARM_TO_LP:
            lp_out.holding_wafer = atr_right_arm.holding_wafer
            lp_out.process_time = self.LP_IN_OUT_PROCESS_TIME
            atr_right_arm.holding_wafer = None
            lp_out.pending_action = None
            atr_right_arm.pending_action = None
            
    def _complete_vtr_action(self, component):
        entry = self.components[ComponentType.ENTRY]
        exit = self.components[ComponentType.EXIT]
        vtr_left_arm = self.components[ComponentType.VTR_LEFT_ARM]
        vtr_right_arm = self.components[ComponentType.VTR_RIGHT_ARM]
        chambers = self.components[ComponentType.CHAMBER]

        vtr_action = component.pending_action

        if vtr_action == VTRAction.ENTRY_TO_LEFT_ARM:
            vtr_left_arm.holding_wafer = entry.holding_wafer
            entry.holding_wafer = None
            entry.process_time = self.ENTRY_EXIT_PROCESS_TIME
            vtr_left_arm.pending_action = None

        elif vtr_action == VTRAction.ENTRY_TO_RIGHT_ARM:
            vtr_right_arm.holding_wafer = entry.holding_wafer
            entry.holding_wafer = None
            entry.process_time = self.ENTRY_EXIT_PROCESS_TIME
            vtr_right_arm.pending_action = None

        elif vtr_action.name.startswith("LEFT_ARM_TO_CHAMBER"):
            chamber_index = int(vtr_action.name.split('_')[-1]) - 1
            chambers[chamber_index].holding_wafer = vtr_left_arm.holding_wafer
            vtr_left_arm.holding_wafer = None
            chambers[chamber_index].process_time = self.getCMBExeTime()
            
            #Chamber Total 수행 시간 파악을 위해 추가
            chambers[chamber_index].cumulative_process_time += chambers[chamber_index].process_time
            chambers[chamber_index].total_cumulative_process_time += chambers[chamber_index].process_time

            chambers[chamber_index].pending_action = None
            vtr_left_arm.pending_action = None

        elif vtr_action.name.startswith("RIGHT_ARM_TO_CHAMBER"):
            chamber_index = int(vtr_action.name.split('_')[-1]) - 1
            chambers[chamber_index].holding_wafer = vtr_right_arm.holding_wafer
            vtr_right_arm.holding_wafer = None
            chambers[chamber_index].process_time = self.getCMBExeTime()
            
            #Chamber Total 수행 시간 파악을 위해 추가
            chambers[chamber_index].cumulative_process_time += chambers[chamber_index].process_time
            chambers[chamber_index].total_cumulative_process_time += chambers[chamber_index].process_time

            chambers[chamber_index].pending_action = None
            vtr_right_arm.pending_action = None

        elif vtr_action.name.startswith("CHAMBER"):
            chamber_index = int(vtr_action.name.split('_')[1]) - 1
            chambers[chamber_index].pending_action = None
            
            if "LEFT_ARM" in vtr_action.name:
                vtr_left_arm.holding_wafer = chambers[chamber_index].holding_wafer
                vtr_left_arm.pending_action = None
            elif "RIGHT_ARM" in vtr_action.name:
                vtr_right_arm.holding_wafer = chambers[chamber_index].holding_wafer
                vtr_right_arm.pending_action = None

            chambers[chamber_index].holding_wafer = None

            # 이때 cumulative_process_time > cleaning limit 초과 시 Cleaning
            if chambers[chamber_index].cumulative_process_time > self.CMD_CLEANING_LIMIT:
                chambers[chamber_index].process_time = self.CMD_CLEANING_TIME
                chambers[chamber_index].cumulative_process_time = 0  

            
        
        elif vtr_action == VTRAction.LEFT_ARM_TO_EXIT:
            exit.holding_wafer = vtr_left_arm.holding_wafer
            vtr_left_arm.pending_action = None
            exit.process_time = self.ENTRY_EXIT_PROCESS_TIME
            vtr_left_arm.holding_wafer = None
        
        elif vtr_action == VTRAction.RIGHT_ARM_TO_EXIT:
            exit.holding_wafer = vtr_right_arm.holding_wafer
            vtr_right_arm.pending_action = None
            exit.process_time = self.ENTRY_EXIT_PROCESS_TIME
            vtr_right_arm.holding_wafer = None


    def _calculate_metrics(self):
        wafer_processed = False
        busy_chamber_cnt = 0
        processed_wait_time = 0

        chambers = self.components[ComponentType.CHAMBER]
        for chamber in chambers:
            if chamber.holding_wafer and chamber.holding_wafer.process_state == WaferProcessState.PROCESSED:
                chamber.processed_wait_time += 1
                processed_wait_time += chamber.processed_wait_time
            else:
                chamber.processed_wait_time = 0

            if chamber.process_time > 0:
                busy_chamber_cnt += 1

        lp_out = self.components[ComponentType.LP_OUT]
        if lp_out.process_time == 0 and lp_out.holding_wafer is not None:
            lp_out.holding_wafer = None
            self.total_processed_wafers += 1
            wafer_processed = True
     
        return wafer_processed, busy_chamber_cnt, processed_wait_time
      

    def _perform_atr_action(self, atr_action: ATRAction):
        entry = self.components[ComponentType.ENTRY]
        exit = self.components[ComponentType.EXIT]
        atr_left_arm = self.components[ComponentType.ATR_LEFT_ARM]
        atr_right_arm = self.components[ComponentType.ATR_RIGHT_ARM]
        aligner = self.components[ComponentType.ALIGNER]
        lp_in = self.components[ComponentType.LP_IN]
        lp_out = self.components[ComponentType.LP_OUT]

        self.atr_last_action = atr_action

        if atr_action == ATRAction.NO_ACTION:
            return 0
        elif atr_action == ATRAction.LP_TO_LEFT_ARM:
            atr_left_arm.pending_action = atr_action
            atr_left_arm.process_time = self.getATRExeTime(atr_left_arm, ATRComponentLocation.LP_IN)
            lp_in.pending_action = atr_action

        elif atr_action == ATRAction.LP_TO_RIGHT_ARM:
            atr_right_arm.pending_action = atr_action
            atr_right_arm.process_time = self.getATRExeTime(atr_right_arm, ATRComponentLocation.LP_IN)
            lp_in.pending_action = atr_action

        elif atr_action == ATRAction.LEFT_ARM_TO_ALIGNER:
            aligner.pending_action = atr_action
            atr_left_arm.pending_action = atr_action
            atr_left_arm.process_time = self.getATRExeTime(atr_left_arm, ATRComponentLocation.ALIGNER)

        elif atr_action == ATRAction.RIGHT_ARM_TO_ALIGNER:
            aligner.pending_action = atr_action
            atr_right_arm.pending_action = atr_action
            atr_right_arm.process_time = self.getATRExeTime(atr_right_arm, ATRComponentLocation.ALIGNER)

        elif atr_action == ATRAction.ALIGNER_TO_LEFT_ARM:
            aligner.pending_action = atr_action
            atr_left_arm.pending_action = atr_action
            atr_left_arm.process_time = self.getATRExeTime(atr_left_arm, ATRComponentLocation.ALIGNER)

        elif atr_action == ATRAction.ALIGNER_TO_RIGHT_ARM:
            aligner.pending_action = atr_action
            atr_right_arm.pending_action = atr_action
            atr_right_arm.process_time = self.getATRExeTime(atr_right_arm, ATRComponentLocation.ALIGNER)

        elif atr_action == ATRAction.LEFT_ARM_TO_ENTRY:
            entry.pending_action = atr_action
            atr_left_arm.pending_action = atr_action
            atr_left_arm.process_time = self.getATRExeTime(atr_left_arm, ATRComponentLocation.ENTRY)

        elif atr_action == ATRAction.RIGHT_ARM_TO_ENTRY:
            entry.pending_action = atr_action
            atr_right_arm.pending_action = atr_action
            atr_right_arm.process_time = self.getATRExeTime(atr_right_arm, ATRComponentLocation.ENTRY)

        elif atr_action == ATRAction.EXIT_TO_LEFT_ARM:
            exit.pending_action = atr_action
            atr_left_arm.pending_action = atr_action
            atr_left_arm.process_time = self.getATRExeTime(atr_left_arm, ATRComponentLocation.EXIT)

        elif atr_action == ATRAction.EXIT_TO_RIGHT_ARM:
            exit.pending_action = atr_action
            atr_right_arm.pending_action = atr_action
            atr_right_arm.process_time = self.getATRExeTime(atr_right_arm, ATRComponentLocation.EXIT)

        elif atr_action == ATRAction.LEFT_ARM_TO_LP:
            lp_out.pending_action = atr_action
            atr_left_arm.pending_action = atr_action
            atr_left_arm.process_time = self.getATRExeTime(atr_left_arm, ATRComponentLocation.LP_OUT)
        elif atr_action == ATRAction.RIGHT_ARM_TO_LP:
            lp_out.pending_action = atr_action
            atr_right_arm.pending_action = atr_action
            atr_right_arm.process_time = self.getATRExeTime(atr_right_arm, ATRComponentLocation.LP_OUT)
            
        return 1     

    def _perform_vtr_action(self, vtr_action: VTRAction):
        entry = self.components[ComponentType.ENTRY]
        exit = self.components[ComponentType.EXIT]
        vtr_left_arm = self.components[ComponentType.VTR_LEFT_ARM]
        vtr_right_arm = self.components[ComponentType.VTR_RIGHT_ARM]
        chambers = self.components[ComponentType.CHAMBER]
	    		
        self.vtr_last_action = vtr_action
              
        if vtr_action == VTRAction.NO_ACTION:
            return 0
        elif vtr_action == VTRAction.ENTRY_TO_LEFT_ARM:
            entry.pending_action = vtr_action
            vtr_left_arm.pending_action = vtr_action
            vtr_left_arm.process_time = self.getVTRExeTime(vtr_left_arm, VTRComponentLocation.ENTRY)

        elif vtr_action == VTRAction.ENTRY_TO_RIGHT_ARM:
            entry.pending_action = vtr_action
            vtr_right_arm.pending_action = vtr_action
            vtr_right_arm.process_time = self.getVTRExeTime(vtr_right_arm, VTRComponentLocation.ENTRY)

        elif vtr_action.name.startswith("LEFT_ARM_TO_CHAMBER"):
            chamber_index = int(vtr_action.name.split('_')[-1]) - 1
            chambers[chamber_index].pending_action = vtr_action
            vtr_left_arm.pending_action = vtr_action
            vtr_left_arm.process_time = self.getVTRExeTime(vtr_left_arm, VTRComponentLocation(chamber_index + 1))
            
        elif vtr_action.name.startswith("RIGHT_ARM_TO_CHAMBER"):
            chamber_index = int(vtr_action.name.split('_')[-1]) - 1
            chambers[chamber_index].pending_action = vtr_action
            vtr_right_arm.pending_action = vtr_action
            vtr_right_arm.process_time = self.getVTRExeTime(vtr_right_arm, VTRComponentLocation(chamber_index + 1))

        elif vtr_action.name.startswith("CHAMBER"):
            chamber_index = int(vtr_action.name.split('_')[1]) - 1
            if "LEFT_ARM" in vtr_action.name:
                chambers[chamber_index].pending_action = vtr_action
                vtr_left_arm.pending_action = vtr_action
                vtr_left_arm.process_time = self.getVTRExeTime(vtr_left_arm, VTRComponentLocation(chamber_index + 1))
            elif "RIGHT_ARM" in vtr_action.name:
                chambers[chamber_index].pending_action = vtr_action
                vtr_right_arm.pending_action = vtr_action
                vtr_right_arm.process_time = self.getVTRExeTime(vtr_right_arm, VTRComponentLocation(chamber_index + 1))

        elif vtr_action == VTRAction.LEFT_ARM_TO_EXIT:
            exit.pending_action = vtr_action
            vtr_left_arm.pending_action = vtr_action
            vtr_left_arm.process_time = self.getVTRExeTime(vtr_left_arm, VTRComponentLocation.EXIT)

        elif vtr_action == VTRAction.RIGHT_ARM_TO_EXIT:
            exit.pending_action = vtr_action
            vtr_right_arm.pending_action = vtr_action
            vtr_right_arm.process_time = self.getVTRExeTime(vtr_right_arm, VTRComponentLocation.EXIT)

        return 1
    
    def perform_action(self, vtr_action: VTRAction, atr_action: ATRAction):
        wafer_processed = False
        busy_chamber_cnt = 0
        processed_wait_time = 0

        self.current_eqp_step += 1
         # Update process times and states
        self._update_component_times()

        # Calculate metrics
        wafer_processed, busy_chamber_cnt, processed_wait_time = self._calculate_metrics()

        vtr_valid = self.check_vtr_action_validity(vtr_action)
        atr_valid = self.check_atr_action_validity(atr_action)

        if not vtr_valid or not atr_valid:
            return -1, wafer_processed, busy_chamber_cnt, processed_wait_time
        
        ret_vtr = self._perform_vtr_action(vtr_action)
        ret_atr = self._perform_atr_action(atr_action)

        return ret_vtr + ret_atr, wafer_processed, busy_chamber_cnt, processed_wait_time

    def render(self, cumulative_reward):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        screen_width, screen_height = 1000, 1100

        # 화면 크기가 변경되었는지 확인
        if not hasattr(self, 'screen') or self.screen.get_size() != (screen_width, screen_height):
            self.screen = pygame.display.set_mode((screen_width, screen_height))

        self.screen.fill((30, 30, 30))  # Dark gray background

        colors = {
            'empty': (200, 200, 200),  # Light gray
            'exist': (100, 200, 100),  # Light green
            'aligned': (100, 200, 200),  # Light blue
            'processed': (200, 100, 200),  # Light purple
            'cleaning': (200, 100, 100),  # Light red
            'font': (255, 255, 255),  # White
            'vac_border': (0, 100, 255),  # Blue for vacuum
            'atm_border': (255, 165, 0)  # Orange for atmospheric
        }

        info_font = pygame.font.Font(None, 24)
        title_font = pygame.font.Font(None, 36)

        # Define positions and sizes
        box_size = (100, 80)
        chamber_size = (80, 80)
        wafer_radius = 30  # Radius of the wafer
        
        start_x, start_y = 50, 100  # Increased start_y to make room for title

        def draw_module(position, component, text, size=box_size, is_chamber=False):
            color = colors['empty']
            wafer_color = None
            wafer_id_text = ""
            process_time_text = str(component.process_time)
            
            if component.holding_wafer is not None:
                wafer = component.holding_wafer
                wafer_id_text = f"ID: {wafer.id}"
                if wafer.process_state == WaferProcessState.BEFORE_PROCESS:
                    wafer_color = colors['exist']
                elif wafer.process_state == WaferProcessState.PROCESSED:
                    wafer_color = colors['processed']
                elif wafer.process_state == WaferProcessState.ALIGNED:
                    wafer_color = colors['aligned']

            if is_chamber and component.process_time > 0 and component.holding_wafer is None:
                color = colors['cleaning']

            pygame.draw.rect(self.screen, color, pygame.Rect(position, size))

            border_color = colors['vac_border'] if component.vac_or_atm == 0 else colors['atm_border']
            pygame.draw.rect(self.screen, border_color, pygame.Rect(position, size), 3)

            # Draw wafer
            if wafer_color:
                wafer_center = (position[0] + size[0] // 2, position[1] + size[1] // 2)
                pygame.draw.circle(self.screen, wafer_color, wafer_center, wafer_radius)

            text_surface = info_font.render(text, True, colors['font'])
            text_rect = text_surface.get_rect(center=(position[0] + size[0] // 2, position[1] - 20))
            self.screen.blit(text_surface, text_rect)

            if wafer_id_text:
                wafer_id_surface = info_font.render(wafer_id_text, True, colors['font'])
                wafer_id_rect = wafer_id_surface.get_rect(center=(position[0] + size[0] // 2, position[1] + size[1] // 2))
                self.screen.blit(wafer_id_surface, wafer_id_rect)

            time_surface = info_font.render(process_time_text, True, colors['font'])
            time_rect = time_surface.get_rect(center=(position[0] + size[0] // 2, position[1] + size[1] + 15))
            self.screen.blit(time_surface, time_rect)

        # Draw title
        title_surface = title_font.render("Equipment Scheduler Visualization", True, colors['font'])
        title_rect = title_surface.get_rect(center=(screen_width // 2, 40))
        self.screen.blit(title_surface, title_rect)

        # Draw Chambers
        chamber_positions = [
            (start_x, start_y), (start_x + 200, start_y), (start_x + 400, start_y),
            (start_x, start_y + 150), (start_x + 200, start_y + 150), (start_x + 400, start_y + 150)
        ]
        for i, pos in enumerate(chamber_positions):
            draw_module(pos, self.components[ComponentType.CHAMBER][i], f"Chamber {i+1}", chamber_size, is_chamber=True)

        # Draw VTR arms
        draw_module((start_x + 100, start_y + 300), self.components[ComponentType.VTR_LEFT_ARM], "VTR Left")
        draw_module((start_x + 300, start_y + 300), self.components[ComponentType.VTR_RIGHT_ARM], "VTR Right")

        # Draw Entry and Exit
        draw_module((start_x, start_y + 450), self.components[ComponentType.ENTRY], "Entry")
        draw_module((start_x + 400, start_y + 450), self.components[ComponentType.EXIT], "Exit")

        # Draw ATR arms
        draw_module((start_x + 100, start_y + 550), self.components[ComponentType.ATR_LEFT_ARM], "ATR Left")
        draw_module((start_x + 300, start_y + 550), self.components[ComponentType.ATR_RIGHT_ARM], "ATR Right")

        # Draw Aligner
        draw_module((start_x + 600, start_y + 450), self.components[ComponentType.ALIGNER], "Aligner")

        # Draw LP_IN and LP_OUT
        draw_module((start_x + 100, start_y + 700), self.components[ComponentType.LP_IN], "LP_IN")
        draw_module((start_x + 300, start_y + 700), self.components[ComponentType.LP_OUT], "LP_OUT")

        # Display information
        info_x, info_y_start = screen_width - 400, 100
        info_gap = 30

        info_texts = [
            f'VTR Action: {self.vtr_last_action.name}',
            f'ATR Action: {self.atr_last_action.name}',
            f'Cumulative Reward: {cumulative_reward:.2f}',
            f'Current Step: {self.current_eqp_step}',
            f'Total Processed Wafers: {self.total_processed_wafers}'
        ]

        for i, text in enumerate(info_texts):
            text_surface = info_font.render(text, True, colors['font'])
            self.screen.blit(text_surface, (info_x, info_y_start + i * info_gap))

        pygame.display.flip()
        self.clock.tick(20)
        
if __name__ == "__main__":


    env = EQP_Scheduler(1000, 10, 300, 305, 30, 200, 300, 10, 10, 5)
    env.reset()

    for _ in range(100000):
        vtr_action = np.random.choice(list(VTRAction))
        atr_action = np.random.choice(list(ATRAction))
        reward, processed, _, _ = env.perform_action(vtr_action, atr_action)
        env.render(0)

