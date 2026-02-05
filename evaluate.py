"""
Standalone evaluation script that can be run directly.
This is a wrapper around src.evaluate that handles imports correctly.
"""

if __name__ == "__main__":
    from src.evaluate import main
    main()
