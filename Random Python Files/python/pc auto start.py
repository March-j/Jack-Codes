from wakepy import set_keepawake, unset_keepawake

set_keepawake(keep_screen_awake=True)
# do stuff that takes long time
unset_keepawake()
