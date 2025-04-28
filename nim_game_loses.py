def minimax(stones, is_computer_turn):
    if stones == 0:
        return 1 if not is_computer_turn else -1

    if is_computer_turn:
        best_score = float('inf')  # <-- NOTE: minimized instead of maximized
        for move in [1, 2, 3]:
            if stones >= move:
                score = minimax(stones - move, False)
                best_score = min(best_score, score)
        return best_score
    else:
        best_score = -float('inf')  # player plays optimally (maximize)
        for move in [1, 2, 3]:
            if stones >= move:
                score = minimax(stones - move, True)
                best_score = max(best_score, score)
        return best_score

def worst_move(stones):
    worst_score = float('inf')
    move_chosen = 1
    for move in [1, 2, 3]:
        if stones >= move:
            score = minimax(stones - move, False)
            if score < worst_score:
                worst_score = score
                move_chosen = move
    return move_chosen

def play_nim():
    stones = int(input("Enter initial number of stones (>= 1): "))
    turn = input("Who plays first? (computer/player): ").strip().lower()

    if turn not in ["computer", "player"]:
        print("Invalid choice.")
        return

    is_computer_turn = (turn == "computer")

    while stones > 0:
        print(f"\nStones left: {stones}")

        if is_computer_turn:
            move = worst_move(stones)
            print(f"Computer removes {move} stone(s).")
        else:
            move = int(input("How many stones do you want to remove (1-3)? "))
            if move not in [1, 2, 3] or move > stones:
                print("Invalid move. Try again.")
                continue

        stones -= move
        is_computer_turn = not is_computer_turn

    if not is_computer_turn:
        print("\nComputer wins!")
    else:
        print("\nYou win!")

if __name__ == "__main__":
    play_nim()
