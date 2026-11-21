def clean_rooms(rooms, position):
    for step in range(2):
        print("Current room:", position)
        if rooms[position] == "Dirty":
            rooms[position] = "Clean"
            print("Action: Suck")
        else:
            print("Room is already clean.")

        if step == 0:
            next_room = "B" if position == "A" else "A"
            print("Action: Move to", next_room)
            position = next_room
    print("Final status:", rooms)

if __name__ == "__main__":
    rooms = {"A": "Dirty", "B": "Dirty"}
    clean_rooms(rooms, "A")
