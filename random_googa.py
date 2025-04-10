def first_fit(items, capacity):
    """
    Implements the first fit bin packing algorithm.

    Args:
        items (list): A list of item sizes.
        capacity (int): The maximum capacity of each bin.

    Returns:
        list: A list of bins, where each bin is a list of items.
    """
    bins = []  # List to hold bins, each bin will be a list of items.

    # Process each item one by one.
    for item in items:
        placed = False
        # Try to place the item in the first bin that has enough remaining capacity.
        for b in bins:
            if sum(b) + item <= capacity:
                b.append(item)
                placed = True
                break
        # If the item does not fit in any existing bin, create a new bin for it.
        if not placed:
            bins.append([item])

    return bins

def main():
    # Define the item set and bin capacity.
    items = [4, 5, 6, 3, 9, 2]
    capacity = 13

    # Call the first_fit algorithm.
    packed_bins = first_fit(items, capacity)

    # Print the bins in the required format.
    for i, bin_items in enumerate(packed_bins, start=1):
        # Format the bin content with comma separated items.
        bin_content = ", ".join(map(str, bin_items))
        print(f"B{i} = {{{bin_content}}}")

if __name__ == "__main__":
    main()
