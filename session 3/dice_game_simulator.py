import random

def play_dice_game():
    num_players = int(input("Enter number of players: "))
    players = [input(f"Enter name for Player {i+1}: ") for i in range(num_players)]
    scores = {player: 0 for player in players}
    last_rolls = {player: 0 for player in players}  # Track last roll for bonus

    rounds = 10
    current_round = 1

    print("\nStarting Dice Game Tournament...\n")

    while current_round <= rounds:
        print(f"\n----- Round {current_round} -----")
        for player in players:
            roll = random.randint(1, 6)
            print(f"{player} rolled: {roll}")

            # Apply game rules
            if roll == 6:
                scores[player] += 10
                if last_rolls[player] == 6:
                    print("Double 6! Extra 5 bonus points!")
                    scores[player] += 5
            elif roll == 1:
                scores[player] -= 5
                print("Rolled a 1! Lost 5 points.")
            else:
                scores[player] += roll

            # Reset score if it goes below 0
            if scores[player] < 0:
                scores[player] = 0
                print("Score dropped below 0. Reset to 0.")

            last_rolls[player] = roll  # Save current roll for next round bonus

            print(f"{player}'s current score: {scores[player]}")

        current_round += 1

    # Find highest scorer using lambda and max()
    winner = max(scores.items(), key=lambda x: x[1])

    print("\n===== Final Scores =====")
    for player, score in scores.items():
        print(f"{player}: {score} points")

    print(f"\nWinner: {winner[0]} with {winner[1]} points!")

# Run the game
play_dice_game()
