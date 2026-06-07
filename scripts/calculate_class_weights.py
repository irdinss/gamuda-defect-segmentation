pixel_counts = {
    "Background": 5828421035,
    "Crack": 121860788,
    "Efflorescence": 20969320,
    "Exposed Rebar": 35950881,
    "Spalling": 211902232,
}


def main():

    total_pixels = sum(
        pixel_counts.values()
    )

    print("\nCLASS WEIGHTS\n")

    for cls_name, count in pixel_counts.items():

        frequency = count / total_pixels

        weight = 1.0 / frequency

        print(
            f"{cls_name:15s}"
            f"{weight:.2f}"
        )


if __name__ == "__main__":
    main()