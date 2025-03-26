# Watch Reinforcement Learning algorithms figure out Tic-Tac-Toe faster than you did!

The game Tic-Tac-Toe is complex enough for many different strategies to be applied, yet simple enough that the entire space of gameplay is enumerable.   From player X's perspective, there are only about 500,000 possible games (a sequence of between 6 and 10 states) that can be played, and they will be played with fewer than 9,000 distinct game board states.  This makes it a good example for watching RL algorithms at work.  

### "Model-based" RL algorithms:

The so-called model-based methods take advantage of this beforehand knowledge of their environments.   In addition to the (usually trivial) initial states, *before* any learning the following are available to them:
*  every possible state (for Tic-Tac-Toe, all possible patterns of X, O, or blank spaces reachable from the empty board through valid moves), 
*  for every non-terminal state $s$:
    * the actions possible from that state: $\text{actions}(s) = \{a_0, ...\}$ (the empty spaces ),
    * the resulting state $s'$ from taking action $a$ in state $s$: $\text{move}(s,a)=s'$ (a new game board with 1 more mark made), and
*  every state's terminal or non-terminal status:  $\text{term}(s)\in\{\text{win-X},\text{win-O},\text{draw},\emptyset \}$.

The **value function** $v_\pi(s)$ is defined as the total (potentially discounted) reward we can expect if we are in state $s$ and following policy $\pi$ from that point forward. [ADD EQN, ADD def of $\pi$, move to DEFs?]

#### 1) Optimally estimating the value function $v_\pi(s)$ of policy $\pi(a,s)$:

Model based algorithms for learning $v(s)$ have access to the game mechanics described above.  These include:
* **Dynamic Programming** (DP):   
    * Set the value of terminal states directly.
    * Compute values for neighboring states (parents) directly by following the state-transition graph and applying the Bellman equation.
* **Iterative Policy Evaluation**:  
    * Set an initial  value function $v_0(s)$ to be zero for all but the terminal states.
    * Iteratively update $v_{t}()$ by evaluating policy $\pi$ at every state $s$, returning action(s) $a$ and following states $\{\text{move}(s,a) \}$ and use the Bellman equation to set $v_{t+1}(s)$. [FIXME]
#### 2) Estimating an optimal policy $\pi_1$ for value function $v_\pi(s)$

After we're confident our value function $v_\pi$ accurately estimates what total reward we can expect following $\pi$, can we learn a new policy, $\pi_1$ that is expected to perform better under the same value function?   In general, setting the new policy $\pi_1$ to recommend actions leading to higher-valued states than what $\pi$ recommends is called **Policy Improvement**.  

In the simple case that we want our policy to always take the best action, i.e. the action leading to the state with highest value, we have:

$$
\pi_1(s) = \argmax_{a \in \text{actions(s)}} \underset  {s' \in \text{move}(s,a)} Ev_{\pi}(s')
$$


The original policy $\pi$ came with no guarantees, so $\pi_1$ might be different. We will always expect higher total reward from following policy $\pi_1$ over $\pi$ if, for every state $s$,  $\pi_1$ recommends an action leading to a state of greater or equal value than what $\pi$ recomends (from the *policy imporovment theorem* [proof?]).
#### 3) Alternating between 1) and 2)

With a new policy $\pi_1$ to follow, the old value function will no longer calculate our expected reward, so we can learn another value fuction $v_{\pi_1}$ from one of the algorithms in section 1.  Iterating this process by alternating between the Policy Evaluation ($\underset E \to$) and Policy Improvement ($\underset I \to$),
$$
\pi_0 \underset E \to v_\pi \underset I \to \pi_1 \underset E \to v_{\pi_1} \underset I \to \pi_2 \underset E \to ...\text{ ,}
$$
until $\pi_i$ converges is **Policy Iteration**.  The Tic-Tac-Toe demo of PI starts with $\pi_0$ as a hard-coded, heuristic algorithm and measures their improvement after each iteration.   

Note:  If we use the simple, optimal policy improvement step of section 2 for $\underset I \to$, this is known as **Value Iteration**.  

### "Model-free" algorithms

Suppose we don't have access to full set of possible game states before we do any learning and have to collect them as we play the game.

For games even slightly more complex than Tic-Tac-Toe, this will likely be the case.  We can determine whether a given state is terminal, but don't have the transition function $\text{move}$ ahead of time and therefore cannot learn a value function from a using one of the methods above.  We have to account for new states as they appear.

#### 4) Monte-carlo 

Initialize with a set of initial states & terminal states (with values set to the reward at that state), then until converged:
* Run many episodes of interaction (games of Tic-Tac-Toe). For each one, record: 
    * the sequence of states and actions,
    * the reward at the end.

* Update the values of $v(s)$ using the Bellman update rule [ADD EQN & PSEUDOCODE]

#### ***COMING SOON:***

#### 5) Q-learning

In-place, stochastic value iteration.

#### 6) Policy gradients
##### 6.5?
#### 7) Proximal Policy Optimization (PPO)

#### 8) Group Level Policy Optimization (GLPO)

# Watch RL algorithms learn Tic-Tac-Toe:

### Generating the Game Tree
The model-based algorithms rely on the full game tree, which can take several seconds to generate.   To save time, the demos will attempt to load a pre-generated tree from the file `game_tree_X.pkl`.  If they don't find it, they will generate it (~ 50 MB).   

Alternatively, to watch it generate with more verbosity, run:
```
ml_demos\rl> python tic_tac_toe.py
```
This will generate all the possible games player X could see, save the results, and then print out their statistics:
```
==========================================================
Saving game tree to cache file:  game_tree_X.pkl
        saved game tree to cache file:  game_tree_X.pkl
==========================================================
Total unique states:  8533
        terminal, X-wins:  942
        terminal, O-wins:  942
        terminal, draw:  32
==========================================================
Games played: 510336

        X goes first:
                X wins: 131184
                O wins: 77904
                Draws: 46080

        O goes first:
                X wins: 77904
                O wins: 131184
                Draws: 46080

        Totals:
                X wins:  209088
                O wins:  209088
                Draws: 92160

Creating Draw States image...
        saved to:  draw_states.png
```
As well as displaying all 32 draw states in the image:

![draw_states.png](draw_states.png)

(lines are green, indicating a draw terminal state.)


---FUTURE WORK PAST HERE---

### The Game Tree App:

This app shows all 8,533 game states in the same window and all edges between them.  To make sense of this:
* mouse-over states to magnify them, their edges, their parents/children up to depth D.  
* Change D with hotkeys.  
* Click to select states, keeping them permanently magnified when the mouse is no longer over them.  Click again to de-select.

Start the app by runing: 
```
ml_demos\rl> python game_graph.py
```

[ADD SCREENSHOT]

## The RL Demo apps

#### Outline for each demo:
1. **Main window**:  Like the Game-Tree app, but with visualization of $v(s)$ of every state, plotted over/near it.  (Possibly only highlighted states if things get crowded).
2. **Function evolution window**:  Like main window, with $v(s)$ being reprsented by a few pixels, in approximately the same locations as the state icons in the main window. Layers with terminal states should put them at opposite horizontal ends, etc.  (not sure if edges will be interesting)
3. **Competition window**:  Have games running continually w/the best RL agent so far vs the starting agent, a heuristic player with varying numbers of rules, a random player, etc.  Plot a history of their win/loss ratios as they learn.

Controls like start/stop/step implemented with hotkeys, press "H" to see.

#### Baseline policies:
1. `Random` player
2. `Heuristic(n)` player, using rules 0 through n and/or defaulting to random:

        1. If there is a winning move, take it.
        2. If the opponent has a winning move, block it.
        3. If the center is open, take it.
        4. If the opponent is in a corner, take the opposite corner.
        5. Take any open corner.

        If no rules apply, take any open space at random.

Note that `Random` is equivalent to `Heuristic(0)`.

Baseline policies are used to "seed" policy improvement algorithms and compete against RL policies to evaluate their progress.        

#### 1) Dynamic Programming

Directly compute the value function of a given policy using DP.  Run the script:

```
ml_demos\rl> python dynamic_prog --seed n [--compare c1 c2 ...] 
```
where the seed policy $\pi_0$ is`Heuristic(n)`, and as training progresses, show its win/loss ratio against the policies `Heuristic(c1)` and `Heuristic(c2)` in the competition window.
