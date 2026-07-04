from clearboard_app import ClearboardApp


def main():
    app = ClearboardApp()
    app.run()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
