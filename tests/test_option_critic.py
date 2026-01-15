import numpy as np
from pathlib import Path
import pytest
import torch

from tests import _PATH_STATEMAPPING
from model.hrl.option_critic.option_critic import Option, OptionCritic
from model.hrl.option_critic.state_manager import StateManager

@pytest.mark.parametrize("index", [32, 64])
@pytest.mark.parametrize("n_states", [32, 64])
@pytest.mark.parametrize("n_actions", [32, 64])
def test_create_options(index: int, n_states: int, n_actions: int) -> None:
    option = Option(index, n_states, n_actions)
    
    assert option.idx == index, "Wrong index number"
    assert option.n_states == n_states, "Wrong number of states"
    assert option.n_actions == n_actions, "Wrong number of actions"
    
    assert option.theta.shape == (n_states, n_actions), "Shape of theta is wrong"
    assert option.upsilon.shape == (n_states,), "Shape of upsilon is wrong"
    
    return

@pytest.mark.parametrize("n_states", [-1, 0])
@pytest.mark.parametrize("n_actions", [-1, 0])
def test_options_wrong_arguments(n_states: int, n_actions: int) -> None:
    index = 0

    with pytest.raises(ValueError):
        Option(index, n_states, n_actions)
    
    return

@pytest.mark.parametrize("temperature", [0.1, 1, 10])
def test_intra_policy(temperature: float) -> None:
    n_states = 2
    n_actions = 2
    option = Option(0, n_states, n_actions)

    index = 0
    option.theta.grad = None
    with torch.no_grad():
        option.theta[index, 0] = 2
    
    option_res = option.pi(index, temperature).detach().numpy()
    
    logits = option.theta[index].detach() / temperature
    result = np.exp(logits.numpy()) / np.sum(np.exp(logits.numpy()))
    
    assert np.allclose(option_res, result), f"Wrong intra-policy updated. Correct: {result}. Got {option_res}"

    index = 1
    option_res = option.pi(index, temperature).detach().numpy()
    
    logits = option.theta[index].detach() / temperature
    result = np.exp(logits.numpy()) / np.sum(np.exp(logits.numpy()))
    
    assert np.allclose(option_res, result), f"Wrong intra-policy updated. Correct: {result}. Got {option_res}"
    
    return

def test_termination_function() -> None:
    n_states = 2
    n_actions = 2
    option = Option(0, n_states, n_actions)
    
    index = 0
    
    option.upsilon.grad = None
    with torch.no_grad():
        option.upsilon += 100
    
    option_res = option.beta(index).detach().numpy()
    
    logits = option.upsilon[index].detach()
    result = 1 / (1 + np.exp(-logits.numpy()))
    
    assert np.allclose(option_res, result), f"Wrong sigmoid calculation. Correct: {result}, got: {option_res}"
    
    return

def test_getters() -> None:
    n_states = 100
    n_actions = 4
    n_options = 2
    n_steps = 500
    state_manager = StateManager(Path(_PATH_STATEMAPPING))
    
    agent = OptionCritic(n_states=n_states, 
                         n_actions=n_actions, 
                         n_options=n_options, 
                         state_manager=state_manager,
                         n_steps=n_steps)
    
    agent.Q_Omega_table = np.random.randn(n_states, n_options)
    agent.Q_U_table = np.random.randn(n_states, n_options, n_actions)
    
    state_idx = np.random.randint(n_states)
    option_idx = np.random.randint(n_options)
    value = agent.get_Q_Omega(state_idx=state_idx, option_idx=option_idx)
    
    assert agent.Q_Omega_table[state_idx, option_idx] == value, "Wrong get output from Q_Omega"
    
    action_idx = np.random.randint(n_actions)
    value = agent.get_Q_U(state_idx=state_idx, option_idx=option_idx, action_idx=action_idx)
    
    assert agent.Q_U_table[state_idx, option_idx, action_idx] == value, "Wrong get output from Q_U"
    
    return

@pytest.mark.parametrize("temperature", [0.1, 1, 10])
def test_setters(temperature: float) -> None:
    n_states = 100
    n_actions = 4
    n_options = 2
    n_steps = 500
    state_manager = StateManager(Path(_PATH_STATEMAPPING))
    
    agent = OptionCritic(n_states=n_states, 
                         n_actions=n_actions, 
                         n_options=n_options, 
                         state_manager=state_manager,
                         n_steps=n_steps)
    
    new_value = 100
    state_idx = np.random.randint(n_states)
    option_idx = np.random.randint(n_options)
    action_idx = np.random.randint(n_actions)
    agent.set_Q_U(state_idx=state_idx, option_idx=option_idx, action_idx=action_idx, new_value=new_value)
    
    assert agent.get_Q_U(state_idx=state_idx, option_idx=option_idx, action_idx=action_idx) == new_value, f"Wrong setter output for Q_U"
    
    agent.set_Q_Omega(state_idx=state_idx, option=agent.options[option_idx], temperature=temperature)
    
    correct_res = agent.options[option_idx].pi(state_idx=state_idx, temperature=temperature) @ agent.get_Q_U(state_idx=state_idx, option_idx=option_idx)

    assert agent.get_Q_Omega(state_idx=state_idx, option_idx=option_idx) == correct_res, f"Wrong setter output for Q_Omega"

    return

@pytest.mark.parametrize("reward", [-10, -1, -0.1, 1, 10])
@pytest.mark.parametrize("terminated", [True, False])
@pytest.mark.parametrize("temperature", [0.1, 1, 10])
def test_options_evaluation(reward: float, terminated: bool, temperature: float) -> None:
    n_states = 100
    n_actions = 4
    n_options = 2
    n_steps = 500
    state_manager = StateManager(Path(_PATH_STATEMAPPING))
    
    agent = OptionCritic(n_states=n_states, 
                         n_actions=n_actions, 
                         n_options=n_options, 
                         state_manager=state_manager,
                         n_steps=n_steps)

    assert np.all(agent.Q_Omega_table.numpy() == np.zeros((n_states, n_options))), "Q_Omega is not all-zeros at initialization"
    assert np.all(agent.Q_U_table.numpy() == np.zeros((n_states, n_options, n_actions))), "Q_U is not all-zeros at initialization"
    
    state_idx = np.random.randint(low = 0, high = n_states - 1)
    option_idx = np.random.randint(low = 0, high = n_options)
    action_idx = np.random.randint(low = 0, high = n_actions)
    agent.options_evaluation(state_idx=state_idx,
                             reward = reward,
                             new_state_idx=state_idx + 1,
                             option=agent.options[option_idx],
                             action=action_idx,
                             terminated=terminated,
                             temperature=temperature)
    
    # NOTE: This would generally not hold, but as this is one step after initialization (zeros), it's cool
    sign_Q_U = np.sign(agent.get_Q_U(state_idx=state_idx, option_idx=option_idx, action_idx=action_idx).numpy())
    sign_r = np.sign(reward)
    
    assert sign_Q_U == sign_r, "The updated Q_U value does not match the sign of the reward"
    
    sign_Q_Omega = np.sign(agent.get_Q_Omega(state_idx=state_idx, option_idx=option_idx).numpy())
    
    assert sign_Q_Omega == sign_r, "The updated Q_Omega value does not match the sign of the reward"
    
    return

@pytest.mark.parametrize("reward", [-10, -1, -0.1, 1, 10])
@pytest.mark.parametrize("terminated", [True, False])
@pytest.mark.parametrize("temperature", [0.1, 1, 10])
@pytest.mark.parametrize("new_value", [-10, 0, 10])
def test_options_improvement(reward: float, terminated: bool, temperature: float, new_value: float) -> None:
    n_states = 100
    n_actions = 4
    n_options = 2
    n_steps = 500
    state_manager = StateManager(Path(_PATH_STATEMAPPING))
    
    agent = OptionCritic(n_states=n_states, 
                         n_actions=n_actions, 
                         n_options=n_options, 
                         state_manager=state_manager,
                         n_steps=n_steps)
    state_idx = np.random.randint(low = 0, high = n_states - 1)
    new_state_idx = state_idx + 1
    option_idx = np.random.randint(low = 0, high = n_options)
    other_option_idx = (n_options - 1) - option_idx
    action_idx = np.random.randint(low = 0, high = n_actions)

    agent.options_evaluation(state_idx=state_idx,
                             reward = reward,
                             new_state_idx=new_state_idx,
                             option=agent.options[option_idx],
                             action=action_idx,
                             terminated=terminated,
                             temperature=temperature)
    
    # NOTE: Q Omega is only updated for the current state and we
    # need an update for the next state to see any difference after
    # option improvement. We do it manually here
    agent.set_Q_U(state_idx=new_state_idx, option_idx=other_option_idx, action_idx=action_idx, new_value=new_value)
    agent.set_Q_Omega(new_state_idx, agent.options[other_option_idx], temperature=temperature)
    
    agent.options_improvement(state_idx=state_idx,
                              new_state_idx=new_state_idx,
                              option = agent.options[option_idx],
                              action=action_idx,
                              temperature=temperature)
    
    # Test for theta
    assert np.all(agent.options[option_idx].theta.grad[state_idx].detach().numpy() != 0), "Some gradients in the option improvement that should be updated, are not"
    
    non_state_idx = np.delete(np.arange(n_states), state_idx)
    assert np.all(agent.options[option_idx].theta.grad[non_state_idx].detach().numpy() == 0), "Some gradients in the option improvement that should NOT be updated, are"
    
    correct_sign = np.ones((n_actions,))
    correct_sign[action_idx] *= np.sign(reward)
    
    non_action_idx = np.delete(np.arange(n_actions), action_idx)
    correct_sign[non_action_idx] *= -np.sign(reward)
    
    assert np.all(np.sign(agent.options[option_idx].theta[state_idx].detach().numpy()) == correct_sign), "The updated theta has a wrong sign somewhere"
    
    # Test for upsilon
    assert np.all(agent.options[option_idx].upsilon.grad[new_state_idx].detach().numpy() != 0), f"Gradients are wrongly updated. new_value: {new_value}"
    
    non_new_state_idx = np.delete(np.arange(n_states), new_state_idx)
    assert np.all(agent.options[option_idx].upsilon.grad[non_new_state_idx].detach().numpy() == 0), f"Gradients are wrongly updated. new_value: {new_value}"
    
    correct_sign = np.sign(new_value) if new_value > 0 else 0
    assert np.all(np.sign(agent.options[option_idx].upsilon[new_state_idx].detach().numpy()) == correct_sign), f"The updated theta has a wrong sign somewhere. new_value: {new_value}"

    return