add_executable(autofocus
  main.cpp
)

target_link_libraries(autofocus PRIVATE ${OpenCV_LIBS} fmt::fmt)
