from enum import Enum
import numpy as np
import pygame

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

class WaferProcessState(Enum):
    BEFORE_PROCESS = 0
    PROCESSED = 1

class ComponentType(Enum):
    ENTRY = 0
    EXIT = 1
    LEFT_ARM = 2
    RIGHT_ARM = 3
    CHAMBER = 4

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

class EQP_Scheduler:
    def __init__(self, MAX_WAFER_NO):
        self.VTR_MinExecutionTime = 2
        self.VTR_MaxExecutionTime = 4
        self.CMB_MinExecutionTime = 7
        self.CMB_MaxExecutionTime = 11
        self.EntryExitWaitTime = 3
        self.components = {
            ComponentType.ENTRY: Component(ComponentType.ENTRY),
            ComponentType.EXIT: Component(ComponentType.EXIT),
            ComponentType.LEFT_ARM: Component(ComponentType.LEFT_ARM),
            ComponentType.RIGHT_ARM: Component(ComponentType.RIGHT_ARM),
            ComponentType.CHAMBER: [Component(ComponentType.CHAMBER, i) for i in range(6)]
        }
        self.wafers = []
        self.next_wafer_id = 1

        self.max_wafer_no = MAX_WAFER_NO
        self.reset()

    def reset(self, seed=None):
        pygame.init()
        self.screen = pygame.display.set_mode((1000, 800))
        self.clock = pygame.time.Clock()
        pygame.display.set_caption('CustomEnv Visualization')

        self.current_eqp_step = 0
        self.total_processed_wafers = 0
        self.last_action = VTRAction.NO_ACTION

        np.random.seed(seed)

        for component_list in self.components.values():
            if isinstance(component_list, list):
                for component in component_list:
                    component.holding_wafer = None
                    component.process_time = 0
            else:
                component_list.holding_wafer = None
                component_list.process_time = 0

        self.wafers = [Wafer(i) for i in range(self.max_wafer_no)]
        self.next_wafer_id = 1

        # Set the first wafer in the entry component
        first_wafer = self.wafers[0]
        self.components[ComponentType.ENTRY].holding_wafer = first_wafer

    def getVTRExeTime(self):
        return np.random.randint(self.VTR_MinExecutionTime, self.VTR_MaxExecutionTime)

    def getCMBExeTime(self):
        return np.random.randint(self.CMB_MinExecutionTime, self.CMB_MaxExecutionTime)

    def check_action_validity(self, vtr_action: VTRAction):
        entry = self.components[ComponentType.ENTRY]
        exit = self.components[ComponentType.EXIT]
        left_arm = self.components[ComponentType.LEFT_ARM]
        right_arm = self.components[ComponentType.RIGHT_ARM]
        chambers = self.components[ComponentType.CHAMBER]

        if vtr_action == VTRAction.NO_ACTION:
            return True
        elif vtr_action == VTRAction.ENTRY_TO_LEFT_ARM or vtr_action == VTRAction.ENTRY_TO_RIGHT_ARM:
            chambers_have_wafer = any(chamber.holding_wafer is not None for chamber in chambers)
            both_arms_empty = left_arm.holding_wafer is None and right_arm.holding_wafer is None
            
            if chambers_have_wafer:
                return (entry.holding_wafer is not None and 
                        left_arm.process_time == 0 and 
                        right_arm.process_time == 0 and 
                        both_arms_empty)
            else:
                if vtr_action == VTRAction.ENTRY_TO_LEFT_ARM:
                    return (left_arm.process_time == 0 and 
                            entry.holding_wafer is not None and 
                            left_arm.holding_wafer is None)
                else:  # ENTRY_TO_RIGHT_ARM
                    return (right_arm.process_time == 0 and 
                            entry.holding_wafer is not None and 
                            right_arm.holding_wafer is None)
        elif vtr_action.name.startswith("LEFT_ARM_TO_CHAMBER"):
            chamber_index = int(vtr_action.name.split('_')[-1]) - 1
            return (left_arm.process_time == 0 and 
                    chambers[chamber_index].process_time == 0 and 
                    left_arm.holding_wafer is not None and 
                    chambers[chamber_index].holding_wafer is None and 
                    left_arm.holding_wafer.process_state == WaferProcessState.BEFORE_PROCESS)
        elif vtr_action.name.startswith("RIGHT_ARM_TO_CHAMBER"):
            chamber_index = int(vtr_action.name.split('_')[-1]) - 1
            return (right_arm.process_time == 0 and 
                    chambers[chamber_index].process_time == 0 and 
                    right_arm.holding_wafer is not None and 
                    chambers[chamber_index].holding_wafer is None and 
                    right_arm.holding_wafer.process_state == WaferProcessState.BEFORE_PROCESS)
        elif vtr_action.name.startswith("CHAMBER"):
            chamber_index = int(vtr_action.name.split('_')[1]) - 1
            if "LEFT_ARM" in vtr_action.name:
                return (left_arm.process_time == 0 and 
                        chambers[chamber_index].process_time == 0 and 
                        left_arm.holding_wafer is None and 
                        chambers[chamber_index].holding_wafer is not None and 
                        chambers[chamber_index].holding_wafer.process_state == WaferProcessState.PROCESSED)
            elif "RIGHT_ARM" in vtr_action.name:
                return (right_arm.process_time == 0 and 
                        chambers[chamber_index].process_time == 0 and 
                        right_arm.holding_wafer is None and 
                        chambers[chamber_index].holding_wafer is not None and 
                        chambers[chamber_index].holding_wafer.process_state == WaferProcessState.PROCESSED)
        elif vtr_action == VTRAction.LEFT_ARM_TO_EXIT:
            return (left_arm.process_time == 0 and 
                    left_arm.holding_wafer is not None and 
                    exit.holding_wafer is None and 
                    left_arm.holding_wafer.process_state == WaferProcessState.PROCESSED)
        elif vtr_action == VTRAction.RIGHT_ARM_TO_EXIT:
            return (right_arm.process_time == 0 and 
                    right_arm.holding_wafer is not None and 
                    exit.holding_wafer is None and 
                    right_arm.holding_wafer.process_state == WaferProcessState.PROCESSED)

        return False

    def perform_action(self, vtr_action: VTRAction):
        entry = self.components[ComponentType.ENTRY]
        exit = self.components[ComponentType.EXIT]
        left_arm = self.components[ComponentType.LEFT_ARM]
        right_arm = self.components[ComponentType.RIGHT_ARM]
        chambers = self.components[ComponentType.CHAMBER]
        wafer_processed = False

        busy_chamber_cnt = 0

        # Update process times
        for chamber in chambers:
            if chamber.process_time > 0:
                chamber.process_time -= 1

                if chamber.process_time == 0 and chamber.holding_wafer is not None:
                    chamber.holding_wafer.process_state = WaferProcessState.PROCESSED
                else:
                    busy_chamber_cnt += 1

        if left_arm.process_time > 0:
            left_arm.process_time -= 1

        if right_arm.process_time > 0:
            right_arm.process_time -= 1
	    
	# Handle entry wait time and supply the next wafer
        if entry.process_time > 0:
            entry.process_time -= 1
            if entry.process_time == 0 and self.next_wafer_id < len(self.wafers):
                next_wafer = self.wafers[self.next_wafer_id]
                self.components[ComponentType.ENTRY].holding_wafer = next_wafer
                self.next_wafer_id += 1
	
	# Handle exit wait time and remove the wafer from exit
        if exit.process_time > 0:
            exit.process_time -= 1
            if exit.process_time == 0 and exit.holding_wafer is not None:
                exit.holding_wafer = None
                self.total_processed_wafers += 1
                wafer_processed = True
				
        self.last_action = vtr_action
        self.current_eqp_step += 1
              
        if vtr_action == VTRAction.NO_ACTION:
            return 0, wafer_processed, busy_chamber_cnt
        elif not self.check_action_validity(vtr_action):
            return -1, wafer_processed, busy_chamber_cnt

        if vtr_action == VTRAction.ENTRY_TO_LEFT_ARM:
            left_arm.holding_wafer = entry.holding_wafer
            entry.holding_wafer = None
            left_arm.process_time = self.getVTRExeTime()
            entry.process_time = self.EntryExitWaitTime  # Set wait time for the next wafer
        elif vtr_action == VTRAction.ENTRY_TO_RIGHT_ARM:
            right_arm.holding_wafer = entry.holding_wafer
            entry.holding_wafer = None
            right_arm.process_time = self.getVTRExeTime()
            entry.process_time = self.EntryExitWaitTime
        elif vtr_action.name.startswith("LEFT_ARM_TO_CHAMBER"):
            chamber_index = int(vtr_action.name.split('_')[-1]) - 1
            chambers[chamber_index].holding_wafer = left_arm.holding_wafer
            left_arm.holding_wafer = None
            left_arm.process_time = self.getVTRExeTime()
            chambers[chamber_index].process_time = self.getCMBExeTime()
        elif vtr_action.name.startswith("RIGHT_ARM_TO_CHAMBER"):
            chamber_index = int(vtr_action.name.split('_')[-1]) - 1
            chambers[chamber_index].holding_wafer = right_arm.holding_wafer
            right_arm.holding_wafer = None
            right_arm.process_time = self.getVTRExeTime()
            chambers[chamber_index].process_time = self.getCMBExeTime()
        elif vtr_action.name.startswith("CHAMBER"):
            chamber_index = int(vtr_action.name.split('_')[1]) - 1
            if "LEFT_ARM" in vtr_action.name:
                left_arm.holding_wafer = chambers[chamber_index].holding_wafer
                chambers[chamber_index].holding_wafer = None
                left_arm.process_time = self.getVTRExeTime()
            elif "RIGHT_ARM" in vtr_action.name:
                right_arm.holding_wafer = chambers[chamber_index].holding_wafer
                chambers[chamber_index].holding_wafer = None
                right_arm.process_time = self.getVTRExeTime()
        elif vtr_action == VTRAction.LEFT_ARM_TO_EXIT:
            exit.holding_wafer = left_arm.holding_wafer
            left_arm.holding_wafer = None
            left_arm.process_time = self.getVTRExeTime()
            exit.process_time = self.EntryExitWaitTime
        elif vtr_action == VTRAction.RIGHT_ARM_TO_EXIT:
            exit.holding_wafer = right_arm.holding_wafer
            right_arm.holding_wafer = None
            right_arm.process_time = self.getVTRExeTime()
            exit.process_time = self.EntryExitWaitTime



        return 1, wafer_processed, busy_chamber_cnt

    def is_impossible_action(self, vtr_action: VTRAction):
        return not self.check_action_validity(vtr_action)

    def render(self, cumulative_reward):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        self.screen.fill((0, 0, 0))

        empty_color = (255, 255, 255)
        exist_color = (0, 255, 0)
        processed_color = (128, 0, 128)
        font_color = (255, 255, 0)

        info_font = pygame.font.Font(None, 24)

        # Define positions
        entry_pos = (100, 250)
        exit_pos = (900, 250)
        left_arm_pos = (450, 250)
        right_arm_pos = (550, 250)
        chamber_positions = [
            (450, 100),  # Chamber 1 (top center)
            (550, 100),  # Chamber 2 (top center)
            (650, 100),  # Chamber 3 (top center)
            (450, 400),  # Chamber 4 (bottom center)
            (550, 400),  # Chamber 5 (bottom center)
            (650, 400)   # Chamber 6 (bottom center)
        ]
        box_size = (50, 50)

        def draw_module(position, component, text):
            color = empty_color
            wafer_id_text = ""
            process_time_text = ""
            if component.holding_wafer is not None:
                wafer = component.holding_wafer
                wafer_id_text = f"ID: {wafer.id}"
                if wafer.process_state == WaferProcessState.BEFORE_PROCESS:
                    color = exist_color
                elif wafer.process_state == WaferProcessState.PROCESSED:
                    color = processed_color

            pygame.draw.rect(self.screen, color, pygame.Rect(position, box_size))

            # Centered text positions
            text_surface = info_font.render(text, True, font_color)
            text_rect = text_surface.get_rect(center=(position[0] + box_size[0] // 2, position[1] - 20))
            self.screen.blit(text_surface, text_rect)

            if wafer_id_text:
                wafer_id_surface = info_font.render(wafer_id_text, True, font_color)
                wafer_id_rect = wafer_id_surface.get_rect(center=(position[0] + box_size[0] // 2, position[1] + box_size[1] // 2))
                self.screen.blit(wafer_id_surface, wafer_id_rect)

            #if component.process_time > 0:
            process_time_text = f"{component.process_time}"
            time_surface = info_font.render(process_time_text, True, font_color)
            time_rect = time_surface.get_rect(center=(position[0] + box_size[0] // 2, position[1] + box_size[1] + 15))
            self.screen.blit(time_surface, time_rect)

        # Draw entry, exit, and arms
        draw_module(entry_pos, self.components[ComponentType.ENTRY], "Entry")
        draw_module(exit_pos, self.components[ComponentType.EXIT], "Exit")
        draw_module(left_arm_pos, self.components[ComponentType.LEFT_ARM], "Left Arm")
        draw_module(right_arm_pos, self.components[ComponentType.RIGHT_ARM], "Right Arm")

        # Draw chambers
        for i, pos in enumerate(chamber_positions):
            draw_module(pos, self.components[ComponentType.CHAMBER][i], f"Chamber {i+1}")

        # Display action, cumulative reward, current step, and total processed wafers
        info_x, info_y_start = 10, 10
        info_gap = 20

        action_enum = VTRAction(self.last_action)
        info_texts = [
            f'Action: {action_enum.name}',
            f'Cumulative Reward: {cumulative_reward}',
            f'Current Step: {self.current_eqp_step}',
            f'Total Processed Wafers: {self.total_processed_wafers}'
        ]

        for i, text in enumerate(info_texts):
            text_surface = info_font.render(text, True, font_color)
            self.screen.blit(text_surface, (info_x, info_y_start + i * info_gap))

        pygame.display.flip()
        self.clock.tick(5)


if __name__ == "__main__":
    env = EQP_Scheduler()
    env.reset()

    for _ in range(1000):
        action = np.random.choice(list(VTRAction))
        reward, processed = env.perform_action(action)
        env.render(env.total_processed_wafers)
    print(env.total_processed_wafers)
